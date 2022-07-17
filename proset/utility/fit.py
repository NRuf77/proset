"""Functions for fitting proset models with good hyperparameters.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks progress of hyperparameter
selection at log level INFO. The invoking application needs to manage log output.
"""

from copy import deepcopy
from enum import Enum
from functools import partial
import logging
from multiprocessing import Pool
import warnings

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_random_state


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAX_SEED = 1e6
LAMBDA_V_RANGE = (1e-6, 1e-1)
LAMBDA_W_RANGE = (1e-9, 1e-4)
NUM_BATCH_GRID = np.arange(11)
SOLVER_FACTR = (1e10, 1e7)
CV_GROUP_FOLD_TOL = 1.2
# warn if the quotient between the size of the largest and smallest cross-validation fold; this is only relevant if
# using the optional cv_groups argument for select_hyperparameters()


class FitMode(Enum):
    """Keep track of which penalty weights are subject to optimization.

    Admissible values are BOTH, LAMBDA_V, LAMBDA_W, and NEITHER.
    """
    BOTH = 0
    LAMBDA_V = 1
    LAMBDA_W = 2
    NEITHER = 3


class Subsampler:  # pylint: disable = too-few-public-methods
    """Class for subsampling data with optional stratification.
    """

    def __init__(self, subsample_rate, stratify, random_state):
        """Initialize subsampler.

        :param subsample_rate: float in (0.0, 1.0); fraction of samples to use for subsampling
        :param stratify: boolean; whether to use stratified subsampling
        :param random_state: an instance of np.random.RandomState, integer, or None; used to initialize the random
            number generator
        """
        if not 0.0 < subsample_rate < 1.0:
            raise ValueError("Parameter subsample_rate must lie in (0.0, 1.0).")
        self._subsample_rate = subsample_rate
        self._stratify = stratify
        self._random_state = check_random_state(random_state)

    def subsample(self, y):
        """Create index vector for subsampling.

        :param y: 1D numpy array
        :return: 1D numpy integer array; indices into y indicating the selected sample
        """
        if not self._stratify:
            return np.sort(self._create_sample_index(y.shape[0]))
        groups = np.unique(y)
        indices = [np.nonzero(y == group)[0] for group in groups]
        indices = [ix[self._create_sample_index(ix.shape[0])] for ix in indices]
        return np.sort(np.hstack(indices))

    def _create_sample_index(self, num_samples):
        """Create index vector for subsample without stratification.

        :param num_samples: positive integer; sample size
        :return: 1D numpy integer array; indices for subsample, not sorted
        """
        return self._random_state.choice(a=num_samples, size=int(np.ceil(self._subsample_rate * num_samples)))


def select_hyperparameters(
        model,
        features,
        target,
        weights=None,
        cv_groups=None,
        transform=None,
        lambda_v_range=None,
        lambda_w_range=None,
        stage_1_trials=50,
        num_batch_grid=None,
        num_folds=5,
        solver_factr=None,
        max_samples=None,
        num_jobs=None,
        random_state=None
):
    """Select hyperparameters for proset model via cross-validation.

    Note: stage 1 uses randomized search in case that a range is given for both lambda_v and lambda_w, sampling values
    uniformly on the log scale. If only one parameter is to be varied, grid search is used instead with values that are
    equidistant on the log scale.

    :param model: an instance of a proset model
    :param features: 2D numpy float array; feature matrix; sparse matrices or infinite/missing values not supported
    :param target: list-like object; target for supervised learning
    :param weights: 1D numpy array of positive floats or None; sample weights used for fitting and scoring; pass None to
        use unit weights
    :param cv_groups: 1D numpy integer array or None; if not None, defines groups of related samples; during
        cross-validation, elements with the same group index (group index + class for classification) are guaranteed to
        stay in one fold; this reduces the risk that idiosyncratic properties of the group are selected as features;
        there should be many groups, all of similar size, or the cross-validation folds will be skewed
    :param transform: sklearn transformer or None; if not None, the transform is applied as part of the model fit to
        normalize features
    :param lambda_v_range: non-negative float, tuple of two non-negative floats, or None; a single value is used as
        penalty for feature weights in all model fits; if a tuple is given, the first value must be strictly less than
        the second and penalties are taken from this range; pass None to use (1e-6, 1e-1)
    :param lambda_w_range: as lambda_v_range above but for the prototype weights and default (1e-9, 1e-4)
    :param stage_1_trials: positive integer; number of penalty combinations to try in stage 1 of fitting
    :param num_batch_grid: 1D numpy array of non-negative integers or None; batch numbers to try in stage 2 of fitting;
        pass None to use np.arange(0, 11)
    :param num_folds: integer greater than 1; number of cross-validation folds to use
    :param solver_factr: non-negative float, tuple of two non-negative floats, or None; a single value is used to set
        solver tolerance for all model fits; if a tuple is given, the first value is used for cross-validation and the
        second for the final model fit; pass None to use (1e10, 1e7)
    :param max_samples: positive integer or None; maximum number of samples included in one batch for fitting; random
        subsampling is stratified by class for a classifier; use this parameter to control memory usage; pass None to
        use all training samples
    :param num_jobs: integer greater than 1 or None; number of jobs to run in parallel; pass None to disable parallel
        execution; for parallel execution, this function must be called inside an 'if __name__ == "__main__"' block to
        avoid spawning infinitely many subprocesses; as numpy usually runs on four cores already, the number of jobs
        should not be greater than one quarter of the available cores
    :param random_state: an instance of numpy.random.RandomState, integer, or None; used as or to initialize a random
        number generator
    :return: dict with the following fields:
        - 'model': an instance of a proset model fitted to all samples using the selected hyperparameters; if a
          transform is provided, the return value is an instance of sklearn.pipeline.Pipeline bundling the transform
          with the proset model; if max_samples is not None, the model's n_iter parameter is set to 1 as it was built
          by adding individual batches using the warm start option
        - 'stage_1': dict with information on parameter search for lambda_v and lambda_w
        - 'stage_2': dict with information on parameter search for number of batches
        - 'sample_ix': list of 1D numpy integer arrays; index vectors of training samples used for each batch; this is
          only provided if max_samples is not None and less than the number of samples in features
    """
    logger.info("Start hyperparameter selection")
    target = check_array(target, dtype=None, ensure_2d=False)
    # leave detailed validation to the model, but numpy-style indexing must be supported
    # noinspection PyTypeChecker
    cv_groups = _process_cv_groups(cv_groups=cv_groups, target=target, classify=is_classifier(model))
    settings = _process_select_settings(
        model=model,
        transform=transform,
        lambda_v_range=lambda_v_range,
        lambda_w_range=lambda_w_range,
        stage_1_trials=stage_1_trials,
        num_batch_grid=num_batch_grid,
        num_folds=num_folds,
        solver_factr=solver_factr,
        max_samples=max_samples,
        num_jobs=num_jobs,
        random_state=random_state
    )
    if settings["fit_mode"] != FitMode.NEITHER:
        logger.info("Execute search stage 1 for penalty weights")
        stage_1 = _execute_stage_1(
            settings=settings, features=features, target=target, weights=weights, cv_groups=cv_groups
        )
    else:
        logger.info("Skip stage 1, penalty weights are fixed")
        stage_1 = _fake_stage_1(settings["lambda_v_range"], settings["lambda_w_range"])
    settings["model"].set_params(
        lambda_v=stage_1["lambda_grid"][stage_1["selected_index"], 0],
        lambda_w=stage_1["lambda_grid"][stage_1["selected_index"], 1]
    )
    if settings["num_batch_grid"].shape[0] > 1:
        logger.info("Execute search stage 2 for number of batches")
        stage_2 = _execute_stage_2(
            settings=settings, features=features, target=target, weights=weights, cv_groups=cv_groups
        )
    else:
        logger.info("Skip stage 2, number of batches is fixed")
        stage_2 = _fake_stage_2(settings["num_batch_grid"])
    settings["model"].set_params(
        n_iter=stage_2["num_batch_grid"][stage_2["selected_index"]],
        solver_factr=settings["solver_factr"][1],
        random_state=random_state
    )
    logger.info("Fit final model with selected parameters")
    model, sample_ix = _make_final_model(
        settings=settings,
        features=features,
        target=target,
        weights=weights
    )
    logger.info("Hyperparameter selection complete")
    result = {"model": model, "stage_1": stage_1, "stage_2": stage_2}
    if sample_ix is not None:
        result["sample_ix"] = sample_ix
    return result


def _process_cv_groups(cv_groups, target, classify):
    """Process information on cross-validation groups.

    :param cv_groups: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param classify: boolean; whether this is a classification or regression problem
    :return: pandas data frame with the following columns or None:
        - index: integer row index
        - cv_group: as input 'cv_groups'
        - target: as input 'target'; only included if 'classify' is True
        If cv_groups is None, this function also returns None.
    """
    if cv_groups is None:
        return None
    if len(cv_groups.shape) != 1:
        raise ValueError("Parameter cv_groups must be a 1D array.")
    if cv_groups.shape[0] != target.shape[0]:
        raise ValueError("Parameter cv_groups must have one element for each sample.")
    if not np.issubdtype(cv_groups.dtype, np.integer):
        raise TypeError("Parameter cv_groups must be integer.")
    cv_groups = pd.DataFrame({"index": np.arange(cv_groups.shape[0]), "cv_group": cv_groups})
    if classify:
        cv_groups["target"] = target
    return cv_groups


def _process_select_settings(
        model,
        transform,
        lambda_v_range,
        lambda_w_range,
        stage_1_trials,
        num_batch_grid,
        num_folds,
        solver_factr,
        num_jobs,
        max_samples,
        random_state
):
    """Validate and prepare settings for select_hyperparameters().

    :param model: see docstring of select_hyperparameters() for details
    :param transform: see docstring of select_hyperparameters() for details
    :param lambda_v_range: see docstring of select_hyperparameters() for details
    :param lambda_w_range: see docstring of select_hyperparameters() for details
    :param stage_1_trials: see docstring of select_hyperparameters() for details
    :param num_batch_grid: see docstring of select_hyperparameters() for details
    :param num_folds: see docstring of select_hyperparameters() for details
    :param solver_factr: see docstring of select_hyperparameters() for details
    :param max_samples: see docstring of select_hyperparameters() for details
    :param num_jobs: see docstring of select_hyperparameters() for details
    :param random_state: see docstring of select_hyperparameters() for details
    :return: dict with the following fields:
        - model: a deep copy of model
        - transform: if transform is not None, a deep copy of transform
        - lambda_v_range: as input or default if input is None
        - lambda_w_range: as input or default if input is None
        - stage_1_trials: as input
        - fit_mode: one value of enum FitMode
        - num_batch_grid: as input or default if input is None
        - splitter: an sklearn splitter for num_fold folds; an instance of StratifiedKFold if model is a classifier, an
          instance of KFold else
        - solver_factr: tuple of two floats; float input is repeated, tuple input kept, None input replaced by default
        - max_samples: as input
        - num_jobs: as input
        - random_state: an instance of numpy.random.RandomState initialized with input random_state
    """
    if lambda_v_range is None:
        lambda_v_range = LAMBDA_V_RANGE  # no need to copy immutable object
    if lambda_w_range is None:
        lambda_w_range = LAMBDA_W_RANGE
    if num_batch_grid is None:
        num_batch_grid = NUM_BATCH_GRID
    if solver_factr is None:
        solver_factr = SOLVER_FACTR
    elif isinstance(solver_factr, float):
        solver_factr = (solver_factr, solver_factr)
    num_batch_grid = num_batch_grid.copy()  # ensure original value is never changed in place
    _check_select_settings(
        lambda_v_range=lambda_v_range,
        lambda_w_range=lambda_w_range,
        stage_1_trials=stage_1_trials,
        num_batch_grid=num_batch_grid,
        solver_factr=solver_factr,
        max_samples=max_samples,
        num_jobs=num_jobs
    )
    model = deepcopy(model)  # do not change original input
    if transform is not None:
        transform = deepcopy(transform)
    fit_mode = _get_fit_mode(lambda_v_range=lambda_v_range, lambda_w_range=lambda_w_range)
    splitter = StratifiedKFold if is_classifier(model) else KFold
    random_state = check_random_state(random_state)
    return {
        "model": model,
        "transform": transform,
        "lambda_v_range": lambda_v_range,
        "lambda_w_range": lambda_w_range,
        "stage_1_trials": stage_1_trials,
        "fit_mode": fit_mode,
        "num_batch_grid": num_batch_grid,
        "splitter": splitter(n_splits=num_folds, shuffle=True, random_state=random_state),
        "solver_factr": solver_factr,
        "max_samples": max_samples,
        "num_jobs": num_jobs,
        "random_state": random_state
    }


# pylint: disable=too-many-branches
def _check_select_settings(
        lambda_v_range,
        lambda_w_range,
        stage_1_trials,
        num_batch_grid,
        solver_factr,
        max_samples,
        num_jobs
):
    """Check input to select_hyperparameters() for consistency.

    :param lambda_v_range: see docstring of select_hyperparameters() for details
    :param lambda_w_range: see docstring of select_hyperparameters() for details
    :param stage_1_trials: see docstring of select_hyperparameters() for details
    :param num_batch_grid: see docstring of select_hyperparameters() for details
    :param solver_factr: tuple of two positive floats
    :param max_samples: see docstring of select_hyperparameters() for details
    :param num_jobs: see docstring of select_hyperparameters() for details
    :return: no return value; raises an exception if an issue is found
    """
    _check_lambda(lambda_v_range, "lambda_v")
    _check_lambda(lambda_w_range, "lambda_w")
    if not np.issubdtype(type(stage_1_trials), np.integer):
        raise TypeError("Parameter stage_1_trials must be integer.")
    if stage_1_trials < 1:
        raise ValueError("Parameter stage_1_trials must be positive.")
    if len(num_batch_grid.shape) != 1:
        raise ValueError("Parameter num_batch_grid must be a 1D array.")
    if not np.issubdtype(num_batch_grid.dtype, np.integer):
        raise TypeError("Parameter num_batch_grid must be an integer array.")
    if np.any(num_batch_grid < 0):
        raise ValueError("Parameter num_batch_grid must not contain negative values.")
    if np.any(np.diff(num_batch_grid) <= 0):
        raise ValueError("Parameter num_batch_grid must contain strictly increasing values.")
    if len(solver_factr) != 2:
        raise ValueError("Parameter solver_factr must have length two if passing a tuple.")
    if solver_factr[0] <= 0.0 or solver_factr[1] <= 0.0:
        raise ValueError("Parameter solver_factr must be positive / have positive elements.")
    if max_samples is not None:
        if not np.issubdtype(type(max_samples), np.integer):
            raise TypeError("Parameter max_samples must be integer if not passing None.")
        if max_samples < 1:
            raise ValueError("Parameter max_samples must be positive if not passing None.")
    if num_jobs is not None and not np.issubdtype(type(num_jobs), np.integer):
        raise ValueError("Parameter num_jobs must be integer if not passing None.")
    if num_jobs is not None and num_jobs < 2:
        raise ValueError("Parameter num_jobs must be greater than 1 if not passing None.")


def _check_lambda(lambda_range, lambda_name):
    """Check that search range for one penalty term is consistent.

    :param lambda_range: non-negative float or tuple of two non-negative floats; fixed penalty or lower and upper bounds
        for search
    :param lambda_name: string; parameter name to show in exception messages
    :return: no return value; raises an exception if an issue is found
    """
    if isinstance(lambda_range, float):
        if lambda_range < 0.0:
            raise ValueError("Parameter {} must not be negative if passing a single value.".format(lambda_name))
    else:
        if len(lambda_range) != 2:
            raise ValueError("Parameter {} must have length two if passing a tuple.".format(lambda_name))
        if lambda_range[0] < 0.0 or lambda_range[1] < 0.0:
            raise ValueError("Parameter {} must not contain negative values if passing a tuple.".format(lambda_name))
        if lambda_range[0] >= lambda_range[1]:
            raise ValueError(
                "Parameter {} must contain strictly increasing values if passing a tuple.".format(lambda_name)
            )


def _get_fit_mode(lambda_v_range, lambda_w_range):
    """Determine fit mode indicator for stage 1.

    :param lambda_v_range: see docstring of select_hyperparameters() for details; None not allowed
    :param lambda_w_range: see docstring of select_hyperparameters() for details; None not allowed
    :return: one element of FitMode
    """
    if isinstance(lambda_v_range, float):
        if isinstance(lambda_w_range, float):
            return FitMode.NEITHER
        return FitMode.LAMBDA_W
    if isinstance(lambda_w_range, float):
        return FitMode.LAMBDA_V
    return FitMode.BOTH


def _execute_stage_1(settings, features, target, weights, cv_groups):
    """Execute search stage 1 for penalty weights.

    :param settings: dict; as return value of process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param weights: see docstring of select_hyperparameters() for details
    :param cv_groups: as return value of _process_cv_groups()
    :return: dict with the following fields:
        - lambda_grid: 2D numpy float array with two columns; values for lambda tried out
        - fit_mode: settings['fit_mode']
        - scores: 1D numpy float array; mean scores corresponding to lambda_grid
        - threshold: best score minus standard deviation for the corresponding parameter combination
        - best_index: integer; index for best score
        - selected_index: integer; index for selected score
    """
    stage_1_plan, stage_1_parameters = _make_stage_1_plan(
        settings=settings,
        features=features,
        target=target,
        weights=weights,
        cv_groups=cv_groups
    )
    stage_1 = _execute_plan(
        plan=stage_1_plan,
        num_jobs=settings["num_jobs"],
        fit_function=_fit_stage_1,
        collect_function=partial(
            _collect_stage_1,
            num_folds=settings["splitter"].n_splits,
            trials=settings["stage_1_trials"]
        )
    )
    return _evaluate_stage_1(
        scores=stage_1,
        parameters=stage_1_parameters,
        fit_mode=settings["fit_mode"]
    )


def _make_stage_1_plan(settings, features, target, weights, cv_groups):
    """Create cross-validation plan for stage 1.

    :param settings: dict; as return value of _process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param weights: see docstring of select_hyperparameters() for details
    :param cv_groups: as return value of _process_cv_groups()
    :return: two lists of dictionaries; elements of the first list have the following fields:
        - model: a deep copy of field "model" from settings
        - transform: a transform as specified in field "transform" of settings, already fitted to features[train_ix, :]
        - features: as input
        - target: as input
        - weights: as input
        - fold: non-negative integer; index of cross_validation fold used for testing
        - train_ix: 1D numpy integer array; indicator vector for the training set
        - trial: non-negative integer; index of parameter combination used for fitting
        - parameters: dict of hyperparameters to be used for fitting; this function specifies lambda_v, lambda_w,
          solver_factr, and random_state
        - subsampler: an instance of class Subsampler or None if settings["max_samples"] is None or the number of
          samples in train_ix does not exceed settings["max_samples"]; this uses stratification if model is a
          classifier; the subsample rate is set to give the desired number of maximum samples (up to rounding errors)
        The second list contains the distinct combinations of parameters lambda_v and lambda_w.
    """
    lambda_v = _sample_lambda(
        lambda_range=settings["lambda_v_range"],
        trials=settings["stage_1_trials"],
        randomize=settings["fit_mode"] == FitMode.BOTH,
        random_state=settings["random_state"]
    )
    lambda_w = _sample_lambda(
        lambda_range=settings["lambda_w_range"],
        trials=settings["stage_1_trials"],
        randomize=settings["fit_mode"] == FitMode.BOTH,
        random_state=settings["random_state"]
    )
    states = _sample_random_state(
        size=settings["stage_1_trials"] * settings["splitter"].n_splits,  random_state=settings["random_state"]
    )
    parameters = [{"lambda_v": lambda_v[i], "lambda_w": lambda_w[i]} for i in range(settings["stage_1_trials"])]
    train_ix = _make_train_ix(features=features, target=target, cv_groups=cv_groups, splitter=settings["splitter"])
    transforms = _make_transforms(features=features, train_ix=train_ix, transform=settings["transform"])
    # pre-fit transformers to avoid refitting for every experiment; transformed features are not stored to avoid memory
    # issues on large problems
    return [{
        "model": deepcopy(settings["model"]),
        "transform": transforms[i],
        "features": features,
        "target": target,
        "weights": weights,
        "fold": i,
        "train_ix": train_ix[i],
        "trial": j,
        "parameters": {
            **parameters[j],
            "solver_factr": settings["solver_factr"][0],
            "random_state": states[i * settings["stage_1_trials"] + j]
        },
        "subsampler": _make_subsampler(
            max_samples=settings["max_samples"],
            num_samples=train_ix[i].shape[0],
            stratify=is_classifier(settings["model"]),
            random_state=states[i * settings["stage_1_trials"] + j]
        )
    } for i in range(settings["splitter"].n_splits) for j in range(settings["stage_1_trials"])], parameters


def _sample_lambda(lambda_range, trials, randomize, random_state):
    """Sample penalty weights for cross-validation.

    :param lambda_range: see parameter lambda_v_range of select_hyperparameters()
    :param trials: see parameter stage_1_trials of select_hyperparameters()
    :param randomize: boolean; whether to use random sampling in case lambda_range is really a range
    :param random_state: an instance of np.random.RandomState
    :return: 1D numpy float array of length trials; penalty weights for cross_validation
    """
    if isinstance(lambda_range, float):
        return lambda_range * np.ones(trials)
    logs = np.log(lambda_range)
    if randomize:
        offset = random_state.uniform(size=trials)
    else:
        offset = np.linspace(0.0, 1.0, trials)
    return np.exp(logs[0] + offset * (logs[1] - logs[0]))


def _sample_random_state(size, random_state):
    """Create multiple random states with distinct seeds.

    :param size: positive integer; number of states to create
    :param random_state: an instance of np.random.RandomState
    :return: list of np.random.RandomState objects
    """
    seeds = (random_state.uniform(size=size) * MAX_SEED).astype(int)
    return [np.random.RandomState(seed) for seed in seeds]


def _make_train_ix(features, target, cv_groups, splitter):
    """Create index vectors of training folds for cross-validation.

    :param features: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param cv_groups: as return value of _process_cv_groups()
    :param splitter: sklearn splitter to use for cross-validation
    :return: list of 1D numpy integer arrays
    """
    if cv_groups is None:
        train_ix = list(splitter.split(X=features, y=target))
    else:
        keep_columns = list(cv_groups.columns)
        keep_columns.remove("index")
        distinct = cv_groups[keep_columns].drop_duplicates()
        if "target" in distinct.columns:
            train_ix = list(splitter.split(X=np.zeros((distinct.shape[0], 0)), y=distinct["target"].values))
        else:
            train_ix = list(splitter.split(X=np.zeros((distinct.shape[0], 0))))
    train_ix = [ix[0] for ix in train_ix]
    if cv_groups is not None:
        # noinspection PyUnboundLocalVariable
        train_ix = [cv_groups.merge(distinct.iloc[ix], on=keep_columns)["index"].values for ix in train_ix]
        sizes = np.array([ix.shape[0] for ix in train_ix])
        if np.max(sizes / np.min(sizes)) > CV_GROUP_FOLD_TOL:
            warnings.warn(" ".join([
                "The quotient between the sizes of the largest and smallest cross-validation folds",
                "is greater than {:0.2f}.".format(CV_GROUP_FOLD_TOL),
                "This can happen if the number of groups defined in cv_groups is too small",
                "or the group sizes are too variable."
            ]), RuntimeWarning)
    return train_ix


def _make_transforms(features, train_ix, transform):
    """Create list of transformers for training data excluding in turn each left-out fold for cross-validation.

    :param features: see docstring of select_hyperparameters() for details
    :param train_ix: as return value of _make_train_ix()
    :param transform: see docstring of select_hyperparameters() for details
    :return: list of fitted transformers or None values if transform is None
    """
    if transform is None:
        return [None] * len(train_ix)
    return [deepcopy(transform).fit(features[ix, :]) for ix in train_ix]


def _make_subsampler(max_samples, num_samples, stratify, random_state):
    """Create subsampler with appropriate training size.

    :param max_samples: docstring of select_hyperparameters() for details
    :param num_samples: positive integer; number of available samples
    :param stratify: boolean, whether to use stratified subsampling
    :param random_state: an instance of np.random.RandomState
    :return: an instance of class Subsampler if max_samples is not None and num_samples > max_samples; None else
    """
    if max_samples is None:
        return None
    subsample_rate = max_samples / num_samples
    if subsample_rate >= 1.0:
        return None
    return Subsampler(subsample_rate=subsample_rate, stratify=stratify, random_state=random_state)


def _execute_plan(plan, num_jobs, fit_function, collect_function):
    """Execute cross-validation plan.

    :param plan: as return value of _make_stage_1_plan
    :param num_jobs: see docstring of select_hyperparameters() for details
    :param fit_function: function for executing model fits
    :param collect_function: function for processing results
    :return: 2D numpy float array; cross-validation scores with one row per fold and one column per trial
    """
    if num_jobs is None:
        return collect_function(list(map(fit_function, plan)))
    with Pool(num_jobs) as pool:
        result = pool.map(fit_function, plan)
    return collect_function(result)


def _fit_stage_1(step):
    """Fit model using one set of hyperparameters for cross-validation for stage 1.

    :param step: dict; as one element of the return value of _make_stage_1_plan()
    :return: triple of two integers and one float; number of fold, number of trial, score on validation fold
    """
    model, validate_ix = _fit_model(step)
    return step["fold"], step["trial"], model.score(
        X=_prepare_features(features=step["features"], sample_ix=validate_ix, transform=step["transform"]),
        y=step["target"][validate_ix],
        sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None
    )


def _fit_model(step):
    """Fit model using one set of hyperparameters.

    :param step: see docstring of _fit_stage_1() for details
    :return: fitted model and 1D numpy bool array indicating the validation fold
    """
    model = step["model"].set_params(**step["parameters"])
    target, use_ix = _get_training_samples(
        target=step["target"], train_ix=step["train_ix"], subsampler=step["subsampler"]
    )
    model.fit(
        X=_prepare_features(features=step["features"], sample_ix=use_ix, transform=step["transform"]),
        y=target,
        sample_weight=step["weights"][use_ix] if step["weights"] is not None else None
    )
    return model, _invert_index(index=step["train_ix"], max_value=step["features"].shape[0])
    # validation fold is not affected by subsampling


def _invert_index(index, max_value):
    """Find complement of integer index.

    :param index: 1D numpy integer array
    :param max_value: positive integer; maximum index value
    :return: 1D numpy integer array, complementary index
    """
    complement = np.ones(max_value, dtype=bool)
    complement[index] = False
    return np.nonzero(complement)[0]


def _get_training_samples(target, train_ix, subsampler):
    """Provide reduced target vector and index for one batch of training data.

    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param train_ix: 1D numpy integer array; indicator vector of training samples
    :param subsampler: an instance of class Subsampler
    :return: two return values:
        - numpy array; reduced target vector
        - 1D numpy integer array; indicator vector for subsampling
    """
    target = target[train_ix]
    if subsampler is not None:
        subset_ix = subsampler.subsample(target)
        target = target[subset_ix]
        train_ix = train_ix[subset_ix]
    return target, train_ix


def _prepare_features(features, sample_ix, transform):
    """Prepare feature matrix for model fitting or scoring.

    :param features: see docstring of select_hyperparameters() for details
    :param sample_ix: 1D numpy integer array with one element per row of features; indicate rows to include in output
    :param transform: a fitted sklearn transformer or None
    :return: feature matrix reduced to relevant samples and transformed if applicable
    """
    features = features[sample_ix, :]
    if transform is not None:
        features = transform.transform(features)
    return features


def _collect_stage_1(results, num_folds, trials):
    """Collect cross-validation results for stage 1.

    :param results: list; elements formatted as return value of _fit_stage_1()
    :param num_folds: see docstring of select_hyperparameters() for details
    :param trials: see parameter stage_1_trials of select_hyperparameters()
    :return: as return value of _execute_plan()
    """
    scores = np.zeros((num_folds, trials))
    for result in results:
        scores[result[0], result[1]] = result[2]
    return scores


def _evaluate_stage_1(scores, parameters, fit_mode):
    """Select hyperparameters for stage 1.

    :param scores: as return value of _execute_plan()
    :param parameters: as second return value of _make_stage_1_plan()
    :param fit_mode: as field "fit_mode" of return value of _process_select_settings()
    :return: as return value of _execute_stage_1()
    """
    stats = _compute_stats(scores)
    lambda_grid = np.array([[p["lambda_v"], p["lambda_w"]] for p in parameters])
    reference = np.sum(np.log(lambda_grid), axis=1)  # this is equivalent to the geometric mean of the penalties
    candidates = np.logical_and(stats["mean"] >= stats["threshold"], reference >= reference[stats["best_index"]])
    # candidate points are all points where (a) the mean score is within one standard error of the optimum (equivalent)
    # and (b) the geometric mean of the two penalties is greater or equal to the value at the optimum (more sparse)
    return {
        "lambda_grid": lambda_grid,
        "fit_mode": fit_mode,
        "scores": stats["mean"],
        "threshold": stats["threshold"],
        "best_index": stats["best_index"],
        "selected_index": np.nonzero(reference == np.max(reference[candidates]))[0][0]
    }


def _compute_stats(scores):
    """Determine optimal parameter combination and threshold from cross-validation scores.

    :param scores: as return value of _execute_plan()
    :return: dict with the following fields:
        - mean: 1D numpy float array; column means of stage
        - best_index: non-negative integer; index of largest mean score
        - threshold: float; largest mean score minus standard deviation for that parameter combination
    """
    mean = np.mean(scores, axis=0)
    best_index = np.argmax(mean)
    return {
        "mean": mean,
        "best_index": best_index,
        "threshold": mean[best_index] - np.std(scores[:, best_index], ddof=1)
        # use variance correction for small sample size
    }


def _fake_stage_1(lambda_v, lambda_w):
    """Generate stage 1 output results for fixed penalty weights.

    :param lambda_v: non-negative float
    :param lambda_w: non-negative float
    :return: as return value of _evaluate_stage_1()
    """
    return {
        "lambda_grid": np.array([[lambda_v, lambda_w]]),
        "fit_mode": FitMode.NEITHER,
        "scores": np.array([np.NaN]),
        "threshold": np.NaN,
        "best_index": 0,
        "selected_index": 0
    }


def _execute_stage_2(settings, features, target, weights, cv_groups):
    """Execute search stage 2 for number of batches.

    :param settings: dict; as return value of process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param weights: see docstring of select_hyperparameters() for details
    :param cv_groups: as return value of _process_cv_groups()
    :return: dict with the following fields:
        - num_batch_grid: as field 'num_batch_grid' of settings
        - scores: 1D numpy float array; mean scores corresponding to num_batch_grid
        - threshold: best score minus standard deviation for the corresponding parameter combination
        - best_index: integer; index for best score
        - selected_index: integer; index for selected score
    """
    stage_2_plan = _make_stage_2_plan(
        settings=settings, features=features, target=target, weights=weights, cv_groups=cv_groups
    )
    stage_2 = _execute_plan(
        plan=stage_2_plan,
        num_jobs=settings["num_jobs"],
        fit_function=_fit_stage_2,
        collect_function=np.vstack
    )
    return _evaluate_stage_2(scores=stage_2, num_batch_grid=settings["num_batch_grid"])


def _make_stage_2_plan(settings, features, target, weights, cv_groups):
    """Create cross-validation plan for stage 2.

    :param settings: dict; as return value of _process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param weights: see docstring of select_hyperparameters() for details
    :param cv_groups: as return value of _process_cv_groups()
    :return: lists of dictionaries; each element has the following fields:
        - model: a deep copy of field "model" from settings
        - transform: a transform as specified in field "transform" of settings, already fitted to features[train_ix, :]
        - features: as input
        - target: as input
        - weights: as input
        - num_batch_grid: as input
        - train_ix: 1D numpy integer array; indicator vector for the training set
        - parameters: dict of hyperparameters to be used for fitting; this function specifies n_iter, solver_factr, and
          random_state
        - subsampler: an instance of class Subsampler or None if settings["max_samples"] is None or the number of
          samples in train_ix does not exceed settings["max_samples"]; this uses stratification if model is a
          classifier; the subsample rate is set to give the desired number of maximum samples (up to rounding errors)
    """
    n_iter = settings["num_batch_grid"][-1]
    states = _sample_random_state(size=settings["splitter"].n_splits, random_state=settings["random_state"])
    train_ix = _make_train_ix(features=features, target=target, cv_groups=cv_groups, splitter=settings["splitter"])
    transforms = _make_transforms(features=features, train_ix=train_ix, transform=settings["transform"])
    # pre-fit transformers to avoid refitting for every experiment; transformed features are not stored to avoid memory
    # issues on large problems
    return [{
        "model": deepcopy(settings["model"]),
        "transform": transforms[i],
        "features": features,
        "target": target,
        "weights": weights,
        "num_batch_grid": settings["num_batch_grid"],
        "train_ix": train_ix[i],
        "parameters": {"n_iter": n_iter, "solver_factr": settings["solver_factr"][0], "random_state": states[i]},
        "subsampler": _make_subsampler(
            max_samples=settings["max_samples"],
            num_samples=train_ix[i].shape[0],
            stratify=is_classifier(settings["model"]),
            random_state=states[i]
        )
    } for i in range(len(train_ix))]


def _fit_stage_2(step):
    """Fit model using one set of hyperparameters for cross-validation for stage 2.

    :param step: dict; as one element of the return value of _make_stage_2_plan()
    :return: 1D numpy float array; scores for different numbers of batches to be evaluated
    """
    if step["subsampler"] is None:
        model, validate_ix = _fit_model(step)
    else:
        model, validate_ix = _fit_with_subsampling(step)
    return model.score(
        X=_prepare_features(features=step["features"], sample_ix=validate_ix, transform=step["transform"]),
        y=step["target"][validate_ix],
        sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None,
        n_iter=step["num_batch_grid"]
    )


def _fit_with_subsampling(step):
    """Fit model with multiple batches using subsampling for every batch.

    :param step: dict; as one element of the return value of _make_stage_2_plan(); field 'subsampler' must not be None
    :return: two return values:
        - fitted proset model object
        - 1D numpy integer array; indicator vector of validation fold
    """
    model = _use_subsampling(
        model=step["model"].set_params(**step["parameters"]),
        transform=step["transform"],
        features=step["features"],
        target=step["target"],
        weights=step["weights"],
        train_ix=step["train_ix"],
        num_batches=step["parameters"]["n_iter"],
        subsampler=step["subsampler"],
        return_ix=False
    )
    return model, _invert_index(index=step["train_ix"], max_value=step["features"].shape[0])
    # validation fold is not affected by subsampling


def _use_subsampling(model, transform, features, target, weights, train_ix, num_batches, subsampler, return_ix):
    """Iteratively add subsets of the training data to a proset model.

    :param model: see return value of _make_stage_2_plan() for details
    :param transform: see return value of _make_stage_2_plan() for details
    :param features: see return value of _make_stage_2_plan() for details
    :param target: see return value of _make_stage_2_plan() for details
    :param weights: see return value of _make_stage_2_plan() for details
    :param train_ix: see return value of _make_stage_2_plan() for details
    :param num_batches: see return value of _make_stage_2_plan() for details
    :param subsampler: see return value of _make_stage_2_plan() for details
    :param return_ix: boolean; whether to return the list of indices for subsampling as second return value
    :return: two return values:
        - model as input, after fitting
        - list of 1D numpy integer arrays; list of index vectors for subsampling at each stage; only provided if
          return_ix is True
    """
    target = target[train_ix]
    model.set_params(n_iter=0)  # initialize model with marginal probabilities
    model.fit(
        X=_prepare_features(features=features, sample_ix=train_ix, transform=transform),
        y=target,
        sample_weight=weights[train_ix] if weights is not None else None,
    )
    model.set_params(n_iter=1)  # now add batches separately
    collect_ix = [] if return_ix else None
    for _ in range(num_batches):
        subset_ix = subsampler.subsample(target)
        use_ix = train_ix[subset_ix]
        if return_ix:
            collect_ix.append(use_ix)
        model.fit(
            X=_prepare_features(features=features, sample_ix=use_ix, transform=transform),
            y=target[subset_ix],
            sample_weight=weights[use_ix] if weights is not None else None,
            warm_start=True  # keep adding batches
        )
    if return_ix:
        return model, collect_ix
    return model


def _evaluate_stage_2(scores, num_batch_grid):
    """Select hyperparameters for stage 2.

    :param scores: as return value of _execute_plan()
    :param num_batch_grid: see docstring of select_hyperparameters() for details
    :return: as return value of _execute_stage_2()
    """
    stats = _compute_stats(scores)
    selected_index = np.nonzero(stats["mean"] >= stats["threshold"])[0][0]
    return {
        "num_batch_grid": num_batch_grid,
        "scores": stats["mean"],
        "threshold": stats["threshold"],
        "best_index": stats["best_index"],
        "selected_index": selected_index
    }


def _fake_stage_2(num_batch_grid):
    """Generate stage 2 output results for fixed number of batches.

    :param num_batch_grid: see docstring of select_hyperparameters() for details
    :return: as return value of _evaluate_stage_2()
    """
    return {
        "num_batch_grid": num_batch_grid,
        "scores": np.array([np.NaN]),
        "threshold": np.NaN,
        "best_index": 0,
        "selected_index": 0
    }


def _make_final_model(
        settings,
        features,
        target,
        weights
):
    """Fit model on all available data with selected hyperparameters.

    :param settings: as return value of _process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: numpy array; target for supervised learning; dimension and data type depend on model
    :param weights: see docstring of select_hyperparameters() for details
    :return: two return values:
        - machine learning model; if settings["transform"] is None, returns settings["model"] after fitting; else,
          returns ans sklearn Pipeline object containing the fitted transform and model
        - list of 1D numpy integer arrays or None; list of index vectors for subsampling at each stage; None if no
          subsampling was used
    """
    if settings["transform"] is not None:
        settings["transform"].fit(features)
    subsampler = _make_subsampler(
        max_samples=settings["max_samples"],
        num_samples=features.shape[0],
        stratify=is_classifier(settings["model"]),
        random_state=settings["random_state"]
    )
    if subsampler is None:
        if settings["transform"] is not None:
            features = settings["transform"].transform(features)
        settings["model"].fit(X=features, y=target, sample_weight=weights)
        sample_ix = None
    else:
        num_batches = settings["model"].get_params()["n_iter"]
        settings["model"], sample_ix = _use_subsampling(
            model=settings["model"],
            transform=settings["transform"],
            features=features,
            target=target,
            weights=weights,
            train_ix=np.arange(features.shape[0]),
            num_batches=num_batches,
            subsampler=subsampler,
            return_ix=True
        )
    if settings["transform"] is not None:
        return Pipeline([("transform", settings["transform"]), ("model", settings["model"])]), sample_ix
    return settings["model"], sample_ix
