"""Functions for fitting proset models with good hyperparameters.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks progress of hyperparameter
selection at log level INFO. The invoking application needs to manage log output.
"""

from copy import deepcopy
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
LOG_START = "Start hyperparameter selection"
LOG_CV = "Select parameters via cross-validation"
LOG_FINAL = "Fit final model with selected parameters"
LOG_DONE = "Hyperparameter selection complete"

MAX_SEED = 1e6
CV_GROUP_FOLD_TOL = 1.2
# warn if the quotient between the size of the largest and smallest cross-validation fold; this is only relevant if
# using the optional cv_groups argument for select_hyperparameters()


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
        lambda_v_grid=np.logspace(-6.0, -1.0, 11),
        max_batches=10,
        num_folds=5,
        solver_factr=(1e10, 1e7),
        max_samples=None,
        num_jobs=None,
        random_state=None
):
    """Select hyperparameters for proset model via cross-validation.

    :param model: an instance of a proset model; hyperparameters that are no subject to cross-validation should already
        be initialized to the desired values
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
    :param lambda_v_grid: non-negative float or 1D numpy array of non-negative floats in ascending order; penalties for
        feature weights to try out
    :param max_batches: positive integer; maximal number of batches of prototypes to try out
    :param num_folds: integer greater than 1; number of cross-validation folds to use
    :param solver_factr: non-negative float, tuple of two non-negative floats, or None; a single value is used to set
        solver tolerance for all model fits; if a tuple is given, the first value is used for cross-validation and the
        second for the final model fit
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
        - 'search': dict with the following fields:
          - cv_results: pandas data frame with columns 'lambda_v', 'num_batches', and 'mean_score' containing all tested
            parameter combinations and mean model scores from cross-validation
          - threshold: best score minus standard deviation for the corresponding parameter combination
          - best_ix: integer; row index for best score in cv_results
          - selected_ix: integer; row index for selected score in cv_results
        - 'sample_ix': list of 1D numpy integer arrays; index vectors of training samples used for each batch; this is
          only provided if max_samples is not None and less than the number of samples in features
    """
    logger.info(LOG_START)
    target, cv_groups, settings = _check_select_input(
        model=model,
        target=target,
        cv_groups=cv_groups,
        transform=transform,
        lambda_v_grid=lambda_v_grid,
        max_batches=max_batches,
        num_folds=num_folds,
        solver_factr=solver_factr,
        max_samples=max_samples,
        num_jobs=num_jobs,
        random_state=random_state
    )
    logger.info(LOG_CV)
    search = _execute_search(settings=settings, features=features, target=target, weights=weights, cv_groups=cv_groups)
    logger.info(LOG_FINAL)
    model, sample_ix = _make_final_model(
        settings=settings, search=search, features=features, target=target, weights=weights
    )
    logger.info(LOG_DONE)
    result = {"model": model, "search": search}
    if sample_ix is not None:
        result["sample_ix"] = sample_ix
    return result


def _check_select_input(
        model,
        target,
        cv_groups,
        transform,
        lambda_v_grid,
        max_batches,
        num_folds,
        solver_factr,
        max_samples,
        num_jobs,
        random_state
):
    """Check and process input for select_hyperparameters().

    :param model: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param cv_groups: see docstring of select_hyperparameters() for details
    :param transform: see docstring of select_hyperparameters() for details
    :param lambda_v_grid: see docstring of select_hyperparameters() for details
    :param max_batches: see docstring of select_hyperparameters() for details
    :param num_folds: see docstring of select_hyperparameters() for details
    :param solver_factr: see docstring of select_hyperparameters() for details
    :param max_samples: see docstring of select_hyperparameters() for details
    :param num_jobs: see docstring of select_hyperparameters() for details
    :param random_state: see docstring of select_hyperparameters() for details
    :return: three return values
        - numpy array with same size and content as input target
        - pandas data frame with the following columns or None (if input cv_groups is None):
          - index: integer row index
          - cv_group: as input cv_groups
          - target: as input target; only included if model is a classifier
        - dict with the following fields:
           - model: a deep copy of model
           - transform: if transform is not None, a deep copy of transform
           - lambda_v_grid: 1D numpy float array; if input lambda_v_grid is a single float, this is wrapped in an array;
             a deep copy of lambda_v_grid otherwise
           - max_batches: as input
           - splitter: an sklearn splitter for num_fold folds; an instance of StratifiedKFold if model is a classifier,
             an instance of KFold else
           - solver_factr: tuple of two floats; if input solver_factr is a single float, this is repeated; as input
             otherwise
           - max_samples: as input
           - num_jobs: as input
           - random_state: an instance of numpy.random.RandomState initialized with input random_state
    """
    target = check_array(target, dtype=None, ensure_2d=False)
    # leave detailed validation to the model, but numpy-style indexing must be supported
    # noinspection PyTypeChecker
    cv_groups = _process_cv_groups(cv_groups=cv_groups, target=target, classify=is_classifier(model))
    settings = _process_select_settings(
        model=model,
        transform=transform,
        lambda_v_grid=lambda_v_grid,
        max_batches=max_batches,
        num_folds=num_folds,
        solver_factr=solver_factr,
        max_samples=max_samples,
        num_jobs=num_jobs,
        random_state=random_state
    )
    return target, cv_groups, settings


def _process_cv_groups(cv_groups, target, classify):
    """Process information on cross-validation groups.

    :param cv_groups: see docstring of select_hyperparameters() for details
    :param target: as first return value of _check_input()
    :param classify: boolean; whether this is a classification or regression problem
    :return: as second return value of _check_input()
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
        lambda_v_grid,
        max_batches,
        num_folds,
        solver_factr,
        num_jobs,
        max_samples,
        random_state
):
    """Validate and collect settings for parameter selection.

    :param model: see docstring of select_hyperparameters() for details
    :param transform: see docstring of select_hyperparameters() for details
    :param lambda_v_grid: see docstring of select_hyperparameters() for details
    :param max_batches: see docstring of select_hyperparameters() for details
    :param num_folds: see docstring of select_hyperparameters() for details
    :param solver_factr: see docstring of select_hyperparameters() for details
    :param max_samples: see docstring of select_hyperparameters() for details
    :param num_jobs: see docstring of select_hyperparameters() for details
    :param random_state: see docstring of select_hyperparameters() for details
    :return: as third return value of _check_select_input()
    """
    model = deepcopy(model)  # do not change original input
    if transform is not None:
        transform = deepcopy(transform)
    if isinstance(lambda_v_grid, float):
        lambda_v_grid = np.array([lambda_v_grid])
    else:
        lambda_v_grid = lambda_v_grid.copy()
    if isinstance(solver_factr, float):
        solver_factr = (solver_factr, solver_factr)
    splitter = StratifiedKFold if is_classifier(model) else KFold
    random_state = check_random_state(random_state)
    _check_select_settings(
        lambda_v_grid=lambda_v_grid,
        max_batches=max_batches,
        solver_factr=solver_factr,
        max_samples=max_samples,
        num_jobs=num_jobs
    )
    return {
        "model": model,
        "transform": transform,
        "lambda_v_grid": lambda_v_grid,
        "max_batches": max_batches,
        "splitter": splitter(n_splits=num_folds, shuffle=True, random_state=random_state),
        "solver_factr": solver_factr,
        "max_samples": max_samples,
        "num_jobs": num_jobs,
        "random_state": random_state
    }


# pylint: disable=too-many-branches
def _check_select_settings(lambda_v_grid, max_batches, solver_factr, max_samples, num_jobs):
    """Check parameters controlling model fit for consistency.

    :param lambda_v_grid: 1D numpy array of non-negative floats in ascending order
    :param max_batches: see docstring of select_hyperparameters() for details
    :param solver_factr: tuple of two positive floats
    :param max_samples: see docstring of select_hyperparameters() for details
    :param num_jobs: see docstring of select_hyperparameters() for details
    :return: no return value; raises an exception if an issue is found
    """
    if len(lambda_v_grid.shape) != 1:
        raise ValueError("Parameter lambda_v_grid must be a 1D array.")
    if np.any(lambda_v_grid < 0.0):
        raise ValueError("Parameter lambda_v_grid must not contain negative values.")
    if np.any(np.diff(lambda_v_grid) <= 0.0):
        raise ValueError("Parameter lambda_v_grid must contain values in ascending order.")
    if not np.issubdtype(type(max_batches), np.integer):
        raise TypeError("Parameter max_batches must be integer.")
    if max_batches < 1:
        raise ValueError("Parameter max_batches must be positive.")
    if len(solver_factr) != 2:
        raise ValueError("Parameter solver_factr must have length two if passing a tuple.")
    if solver_factr[0] <= 0.0 or solver_factr[1] <= 0.0:
        raise ValueError("Parameter solver_factr must be positive / have positive elements.")
    if max_samples is not None:
        if not np.issubdtype(type(max_samples), np.integer):
            raise TypeError("Parameter max_samples must be integer if not passing None.")
        if max_samples < 1:
            raise ValueError("Parameter max_samples must be positive if not passing None.")
    if num_jobs is not None:
        if not np.issubdtype(type(num_jobs), np.integer):
            raise TypeError("Parameter num_jobs must be integer if not passing None.")
        if num_jobs < 2:
            raise ValueError("Parameter num_jobs must be greater than 1 if not passing None.")


def _execute_search(settings, features, target, weights, cv_groups):
    """Select model hyperparameters via cross-validation.

    :param settings: dict; as third return value of _check_select_input()
    :param features: see docstring of select_hyperparameters() for details
    :param target: as first return value of _check_select_input()
    :param weights: see docstring of select_hyperparameters() for details
    :param cv_groups: as second return value of _check_select_input()
    :return: as value under key 'search' in return value of select_hyperparameters()
    """
    plan = _make_plan(settings=settings, features=features, target=target, weights=weights, cv_groups=cv_groups)
    cv_results = _execute_plan(
        plan=plan,
        num_folds=settings["splitter"].n_splits,
        num_trials=settings["lambda_v_grid"].shape[0],
        num_jobs=settings["num_jobs"]
    )
    return _evaluate_search(cv_results=cv_results, settings=settings)


def _make_plan(settings, features, target, weights, cv_groups):
    """Create plan for parameter search via cross-validation.

    :param settings: dict; as third return value of _check_select_input()
    :param features: see docstring of select_hyperparameters() for details
    :param target: as first return value of _check_select_input()
    :param weights: see docstring of select_hyperparameters() for details
    :param cv_groups: as second return value of _check_select_input()
    :return: list of dicts, each with the following fields:
        - model: a deep copy of settings["model"] with updated hyperparameters
        - transform: a transform as specified in field "transform" of settings, already fitted to features[train_ix, :];
          None if no transform is specified
        - features: as input
        - target: as input
        - weights: as input
        - trial: non-negative integer; index into settings["lambda_v_grid"] indicating which value is used by model
        - fold: non-negative integer; index of cross_validation fold used for testing
        - train_ix: 1D numpy integer array; indicator vector for the training set
        - subsampler: an instance of class Subsampler or None if settings["max_samples"] is None or the number of
          samples in train_ix does not exceed settings["max_samples"]; this uses stratification if model is a
          classifier; the subsample rate is set to give the desired number of maximum samples (up to rounding errors)
    """
    train_folds = _make_train_folds(
        features=features, target=target, cv_groups=cv_groups, splitter=settings["splitter"]
    )
    transforms = _make_transforms(features=features, train_folds=train_folds, transform=settings["transform"])
    # pre-fit transformers to avoid refitting for every experiment; transformed features are not stored to avoid memory
    # issues on large problems
    plan = []
    for i, lambda_v in enumerate(settings["lambda_v_grid"]):
        for j, train_ix in enumerate(train_folds):
            random_state = _sample_random_state(random_state=settings["random_state"])
            model = deepcopy(settings["model"])
            model.set_params(n_iter=settings["max_batches"], lambda_v=lambda_v, random_state=random_state)
            plan.append({
                "model": model,
                "transform": transforms[j],
                "features": features,
                "target": target,
                "weights": weights,
                "trial": i,
                "fold": j,
                "train_ix": train_ix,
                "subsampler": _make_subsampler(
                    max_samples=settings["max_samples"],
                    num_samples=train_ix.shape[0],
                    stratify=is_classifier(model),
                    random_state=random_state
                )
            })
    return plan


def _sample_random_state(random_state):
    """Create independent random state with distinct seed from existing state.

    :param random_state: an instance of np.random.RandomState
    :return: another instance np.random.RandomState
    """
    return np.random.RandomState(int(random_state.uniform() * MAX_SEED))


def _make_train_folds(features, target, cv_groups, splitter):
    """Create index vectors of training folds for cross-validation.

    :param features: see docstring of select_hyperparameters() for details
    :param target: as first return value of _check_select_input()
    :param cv_groups: as second return value of _check_select_input()
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


def _make_transforms(features, train_folds, transform):
    """Create list of transformers for training data excluding in turn each left-out fold for cross-validation.

    :param features: see docstring of select_hyperparameters() for details
    :param train_folds: as return value of _make_train_folds()
    :param transform: see docstring of select_hyperparameters() for details
    :return: list of fitted transformers or None values if transform is None
    """
    if transform is None:
        return [None] * len(train_folds)
    return [deepcopy(transform).fit(features[train_ix, :]) for train_ix in train_folds]


def _make_subsampler(max_samples, num_samples, stratify, random_state):
    """Create subsampler with appropriate training size.

    :param max_samples: see docstring of select_hyperparameters() for details
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


def _execute_plan(plan, num_folds, num_trials, num_jobs):
    """Execute cross-validation plan.

    :param plan: as return value of _make_plan()
    :param num_folds: see docstring of select_hyperparameters() for details
    :param num_trials: positive integer; number of different values for lambda_v to be tested
    :param num_jobs: see docstring of select_hyperparameters() for details
    :return: 2D numpy float array; matrix of cross-validation scores with one row per experiment (number of batches
        changing first) and on column per cross-validation fold
    """
    if num_jobs is None:
        step_results = list(map(_fit_step, plan))
    else:
        with Pool(num_jobs) as pool:
            step_results = pool.map(_fit_step, plan)
    return _collect_cv_results(step_results=step_results, num_folds=num_folds, num_trials=num_trials)


def _fit_step(step):
    """Perform a single model fit for cross-validation.

    :param step: a single list entry from the return value if _make_plan()
    :return: three return values:
        - non-negative integer; index of value for lambda_v used
        - non-negative integer; index of cross-validation fold used for scoring
        - scores on the validation fold for each number of batches from 0 to the maximum
    """
    num_batches = step["model"].n_iter
    if step["subsampler"] is None:
        model = step["model"].fit(
            X=_prepare_features(features=step["features"], sample_ix=step["train_ix"], transform=step["transform"]),
            y=step["target"][step["train_ix"]],
            sample_weight=step["weights"][step["train_ix"]] if step["weights"] is not None else None
        )
    else:
        model = _fit_with_subsampling(
            model=step["model"],
            transform=step["transform"],
            features=step["features"],
            target=step["target"],
            weights=step["weights"],
            train_ix=step["train_ix"],
            num_batches=num_batches,
            subsampler=step["subsampler"],
            return_ix=False
        )[0]
    validate_ix = _invert_index(index=step["train_ix"], max_value=step["target"].shape[0])
    # validation fold is not affected by subsampling
    scores = model.score(
        X=_prepare_features(features=step["features"], sample_ix=validate_ix, transform=step["transform"]),
        y=step["target"][validate_ix],
        sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None,
        n_iter=np.arange(num_batches + 1)
    )
    return step["trial"], step["fold"], scores


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


def _fit_with_subsampling(model, transform, features, target, weights, train_ix, num_batches, subsampler, return_ix):
    """Iteratively add subsets of the training data to a proset model.

    :param model: see return value of _make_plan() for details
    :param transform: see return value of _make_plan() for details
    :param features: see return value of _make_plan() for details
    :param target: see return value of _make_plan() for details
    :param weights: see return value of _make_plan() for details
    :param train_ix: see return value of _make_plan() for details
    :param num_batches: see return value of _make_plan() for details
    :param subsampler: see return value of _make_plan() for details
    :return: two return values:
        - model as input, after fitting
        - list of 1D numpy integer arrays; list of index vectors for subsampling at each stage
    """
    target = target[train_ix]
    model.set_params(n_iter=0)  # initialize model with marginal probabilities
    model.fit(
        X=_prepare_features(features=features, sample_ix=train_ix, transform=transform),
        y=target,
        sample_weight=weights[train_ix] if weights is not None else None,
    )
    model.set_params(n_iter=1)  # now add batches separately
    collect_ix = []
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
    return model, collect_ix


def _invert_index(index, max_value):
    """Find complement of integer index.

    :param index: 1D numpy integer array
    :param max_value: positive integer; maximum index value
    :return: 1D numpy integer array, complementary index
    """
    complement = np.ones(max_value, dtype=bool)
    complement[index] = False
    return np.nonzero(complement)[0]


def _collect_cv_results(step_results, num_folds, num_trials):
    """Convert list of cross-validation scores to matrix.

    :param step_results: list of tuples; each tuple as return values from one call to _fit_step()
    :param num_folds: see docstring of select_hyperparameters() for details
    :param num_trials: see docstring of _evaluate_plan() for details
    :return: as return value of _execute_plan()
    """
    max_batches = step_results[0][2].shape[0]
    # this is actually the maximum plus 1 as model fit starts with 0 batches, but the math works out
    cv_result = np.zeros((num_trials * max_batches, num_folds), dtype=float)
    for result in step_results:
        from_ix = result[0] * max_batches
        to_ix = (result[0] + 1) * max_batches
        cv_result[from_ix:to_ix, result[1]] = result[2]
    return cv_result


def _evaluate_search(cv_results, settings):
    """Select hyperparameters based on cross-validation results.

    :param cv_results: as return value of _execute_plan()
    :param settings: as third return value of _check_select_input()
    :return: as return value of _execute_search()
    """
    pd_results = pd.DataFrame({
        "lambda_v": np.repeat(settings["lambda_v_grid"], settings["max_batches"] + 1),
        "num_batches": np.tile(np.arange(settings["max_batches"] + 1), settings["lambda_v_grid"].shape[0]),
        "mean_score": np.mean(cv_results, axis=1)
    })
    best_ix = np.argmax(pd_results["mean_score"].values)
    threshold = pd_results["mean_score"].iloc[best_ix] - np.std(cv_results[best_ix, :], ddof=1)
    candidates = pd_results["mean_score"].values >= threshold
    max_lambda = np.max(pd_results["lambda_v"].values[candidates])
    candidates = np.logical_and(candidates, pd_results["lambda_v"].values == max_lambda)
    min_batches = np.min(pd_results["num_batches"].values[candidates])
    candidates = np.logical_and(candidates, pd_results["num_batches"].values == min_batches)
    selected_ix = np.nonzero(candidates)[0][0]
    return {
        "cv_results": pd_results,
        "threshold": threshold,
        "best_ix": best_ix,
        "selected_ix": selected_ix
    }


def _make_final_model(settings, search, features, target, weights):
    """Fit model on all available data with selected hyperparameters.

    :param settings: as third return value of _check_select_input()
    :param search: as return value of _execute_search()
    :param features: see docstring of select_hyperparameters() for details
    :param target: as first return value of _check_select_input()
    :param weights: see docstring of select_hyperparameters() for details
    :return: two return values:
        - machine learning model; if settings["transform"] is None, returns settings["model"] after fitting; else,
          returns ans sklearn Pipeline object containing the fitted transform and model
        - list of 1D numpy integer arrays or None; list of index vectors for subsampling at each stage; None if no
          subsampling was used
    """
    model = settings["model"]
    model.set_params(
        n_iter=search["cv_results"]["num_batches"].iloc[search["selected_ix"]],
        lambda_v=search["cv_results"]["lambda_v"].iloc[search["selected_ix"]],
        random_state=settings["random_state"]
    )
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
        model.fit(X=features, y=target, sample_weight=weights)
        sample_ix = None
    else:
        num_batches = settings["model"].n_iter
        model, sample_ix = _fit_with_subsampling(
            model=model,
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
        model = Pipeline([("transform", settings["transform"]), ("model", model)])
    return model, sample_ix
