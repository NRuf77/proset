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

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAX_SEED = 1e6
LAMBDA_V_RANGE = (1e-6, 1e-1)
LAMBDA_W_RANGE = (1e-9, 1e-4)
NUM_BATCH_GRID = np.arange(11)


class FitMode(Enum):
    """Keep track of which penalty weights are subject to optimization.

    Admissible values are BOTH, LAMBDA_V, LAMBDA_W, and NEITHER.
    """
    BOTH = 0
    LAMBDA_V = 1
    LAMBDA_W = 2
    NEITHER = 3


def select_hyperparameters(
        model,
        features,
        target,
        weights=None,
        transform=None,
        lambda_v_range=None,
        lambda_w_range=None,
        stage_1_trials=50,
        num_batch_grid=None,
        num_folds=5,
        num_jobs=None,
        random_state=None
):  # pragma: no cover
    """Select hyperparameters for proset model via cross-validation.

    Note: stage 1 uses randomized search in case that a range is given for both lambda_v and lambda_w, sampling values
    uniformly on the log scale. If only one parameter is to be varied, grid search is used instead with values that are
    equidistant on the log scale.

    :param model: an instance of a proset model
    :param features: 2D numpy float array; feature matrix; sparse matrices or infinite/missing values not supported
    :param target: list-like object; target for supervised learning
    :param weights: 1D numpy array of positive floats or None; sample weights used for fitting and scoring; pass None to
        use unit weights
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
    :param num_jobs: integer greater than 1 or None; number of jobs to run in parallel; pass None to disable parallel
        execution; for parallel execution, this function must be called inside an 'if __name__ == "__main__"' block to
        avoid spawning infinitely many subprocesses; as numpy usually runs on four cores already, the number of jobs
        should not be greater than one quarter of the available cores
    :param random_state: an instance of numpy.random.RandomState, integer, or None; used as or to initialize a random
        number generator
    :return: dict with the following fields:
        - 'model': an instance of a proset model fitted to all samples using the selected hyperparameters; if a
          transform is provided, the return value is an instance of sklearn.pipeline.Pipeline bundling the transform
          with the proset model
        - 'stage_1': dict with information on parameter search for lambda_v and lambda_w
        - 'stage_2': dict with information on parameter search for number of batches
    """
    logger.info("Start hyperparameter selection")
    settings = _process_select_settings(
        model=model,
        transform=transform,
        lambda_v_range=lambda_v_range,
        lambda_w_range=lambda_w_range,
        stage_1_trials=stage_1_trials,
        num_batch_grid=num_batch_grid,
        num_folds=num_folds,
        num_jobs=num_jobs,
        random_state=random_state
    )
    if settings["fit_mode"] != FitMode.NEITHER:
        logger.info("Execute search stage 1 for penalty weights")
        stage_1 = _execute_stage_1(settings=settings, features=features, target=target, weights=weights)
    else:
        logger.info("Skip stage 1, penalty weights are fixed")
        stage_1 = _fake_stage_1(settings["lambda_v_range"], settings["lambda_w_range"])
    settings["model"].set_params(**{
        settings["prefix"] + "lambda_v": stage_1["lambda_grid"][stage_1["selected_index"], 0],
        settings["prefix"] + "lambda_w": stage_1["lambda_grid"][stage_1["selected_index"], 1]
    })
    if settings["num_batch_grid"].shape[0] > 1:
        logger.info("Execute search stage 2 for number of batches")
        stage_2 = _execute_stage_2(settings=settings, features=features, target=target, weights=weights)
    else:
        logger.info("Skip stage 2, number of batches is fixed")
        stage_2 = _fake_stage_2(settings["num_batch_grid"])
    settings["model"].set_params(**{
        settings["prefix"] + "n_iter": stage_2["num_batch_grid"][stage_2["selected_index"]],
        settings["prefix"] + "random_state": random_state
    })
    logger.info("Fit final model with selected parameters")
    settings["model"].fit(**{"X": features, "y": target, settings["prefix"] + "sample_weight": weights})
    logger.info("Hyperparameter selection complete")
    return {"model": settings["model"], "stage_1": stage_1, "stage_2": stage_2}


def _process_select_settings(
        model,
        transform,
        lambda_v_range,
        lambda_w_range,
        stage_1_trials,
        num_batch_grid,
        num_folds,
        num_jobs,
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
    :param num_jobs: see docstring of select_hyperparameters() for details
    :param random_state: see docstring of select_hyperparameters() for details
    :return: dict with the following fields:
        - model: if transformer is None, a deep copy of the input model; else, an sklearn pipeline combining deep copies
          of the transformer and model
        - prefix: string; "" if transformer is None, else "model__"; use this to prefix parameters for the proset model
          in case they have to be processed by the pipeline
        - lambda_v_range: as input or default if input is None
        - lambda_w_range: as input or default if input is None
        - stage_1_trials: as input
        - fit_mode: one value of enum FitMode
        - num_batch_grid: as input or default if input is None
        - splitter: an sklearn splitter for num_fold folds; stratified K-fold splitter if model is a classifier,
          ordinary K-fold else
        - num_jobs: as input
        - random_state: an instance of numpy.random.RandomState initialized with input random_state
    """
    if lambda_v_range is None:
        lambda_v_range = LAMBDA_V_RANGE  # no need to copy immutable object
    if lambda_w_range is None:
        lambda_w_range = LAMBDA_W_RANGE
    if num_batch_grid is None:
        num_batch_grid = NUM_BATCH_GRID
    num_batch_grid = num_batch_grid.copy()
    _check_select_settings(
        lambda_v_range=lambda_v_range,
        lambda_w_range=lambda_w_range,
        stage_1_trials=stage_1_trials,
        num_batch_grid=num_batch_grid,
        num_jobs=num_jobs
    )
    model = deepcopy(model)  # do not change original input
    if transform is not None:
        model = Pipeline(steps=[("transform", deepcopy(transform)), ("model", model)])
        prefix = "model__"
    else:
        prefix = ""
    fit_mode = _get_fit_mode(lambda_v_range=lambda_v_range, lambda_w_range=lambda_w_range)
    splitter = StratifiedKFold if is_classifier(model) else KFold
    random_state = check_random_state(random_state)
    return {
        "model": model,
        "prefix": prefix,
        "lambda_v_range": lambda_v_range,
        "lambda_w_range": lambda_w_range,
        "stage_1_trials": stage_1_trials,
        "fit_mode": fit_mode,
        "num_batch_grid": num_batch_grid,
        "splitter": splitter(n_splits=num_folds, shuffle=True, random_state=random_state),
        "num_jobs": num_jobs,
        "random_state": random_state
    }


# pylint: disable=too-many-branches
def _check_select_settings(lambda_v_range, lambda_w_range, stage_1_trials, num_batch_grid, num_jobs):
    """Check input to select_hyperparameters() for consistency.

    :param lambda_v_range: see docstring of select_hyperparameters() for details
    :param lambda_w_range: see docstring of select_hyperparameters() for details
    :param stage_1_trials: see docstring of select_hyperparameters() for details
    :param num_batch_grid: see docstring of select_hyperparameters() for details
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
    if isinstance(lambda_range, np.float):
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
    if isinstance(lambda_v_range, np.float):
        if isinstance(lambda_w_range, np.float):
            return FitMode.NEITHER
        return FitMode.LAMBDA_W
    if isinstance(lambda_w_range, np.float):
        return FitMode.LAMBDA_V
    return FitMode.BOTH


def _execute_stage_1(settings, features, target, weights):  # pragma: no cover
    """Execute search stage 1 for penalty weights.

    :param settings: dict; as return value of process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param weights: see docstring of select_hyperparameters() for details
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
        weights=weights
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


def _make_stage_1_plan(settings, features, target, weights):
    """Create cross-validation plan for stage 1.

    :param settings: dict; as return value of _process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param weights: see docstring of select_hyperparameters() for details
    :return: two lists of dictionaries; elements of the first list have the following fields:
        - model: as field "model" from settings
        - features: as input
        - target: as input
        - weights: as input
        - prefix: as field "model" from settings
        - fold: non-negative integer; index of cross_validation fold used for testing
        - train_ix: 1D numpy boolean array; indicator vector for the training set
        - trial: non-negative integer; index of parameter combination used for fitting
        - parameters: dict of hyperparameters to be used for fitting; this function specifies lambda_v, lambda_w, and
          random_state
        The second list contains the distinct combinations of parameters lambda_v and lambda_w.
    """
    lambda_v = _sample_lambda(
        lambda_range=settings["lambda_v_range"],
        trials=settings["stage_1_trials"],
        do_randomize=settings["fit_mode"] == FitMode.BOTH,
        random_state=settings["random_state"]
    )
    lambda_w = _sample_lambda(
        lambda_range=settings["lambda_w_range"],
        trials=settings["stage_1_trials"],
        do_randomize=settings["fit_mode"] == FitMode.BOTH,
        random_state=settings["random_state"]
    )
    states = _sample_random_state(
        size=settings["stage_1_trials"] * settings["splitter"].n_splits,  random_state=settings["random_state"]
    )
    parameters = [{"lambda_v": lambda_v[i], "lambda_w": lambda_w[i]} for i in range(settings["stage_1_trials"])]
    train_ix = _make_train_ix(features=features, target=target, splitter=settings["splitter"])
    return [{
        "model": deepcopy(settings["model"]),
        "features": features,
        "target": target,
        "weights": weights,
        "prefix": settings["prefix"],
        "fold": i,
        "train_ix": train_ix[i],
        "trial": j,
        "parameters": {**parameters[j], "random_state": states[i * settings["stage_1_trials"] + j]}
    } for i in range(settings["splitter"].n_splits) for j in range(settings["stage_1_trials"])], parameters


def _sample_lambda(lambda_range, trials, do_randomize, random_state):
    """Sample penalty weights for cross-validation.

    :param lambda_range: see parameter lambda_v_range of select_hyperparameters()
    :param trials: see parameter stage_1_trials of select_hyperparameters()
    :param do_randomize: boolean; whether to use random sampling in case lambda_range is really a range
    :param random_state: an instance of np.random.RandomState
    :return: 1D numpy float array of length trials; penalty weights for cross_validation
    """
    if isinstance(lambda_range, np.float):
        return lambda_range * np.ones(trials)
    logs = np.log(lambda_range)
    if do_randomize:
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


def _make_train_ix(features, target, splitter):
    """Create index vectors of training folds for cross-validation.

    :param features: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param splitter: sklearn splitter to use for cross-validation
    :return: list of 1D numpy boolean arrays
    """
    folds = list(splitter.split(X=features, y=target))
    train_ix = []
    for fold in folds:
        new_ix = np.zeros(features.shape[0], dtype=bool)
        new_ix[fold[0]] = True
        train_ix.append(new_ix)
    return train_ix


def _execute_plan(plan, num_jobs, fit_function, collect_function):  # pragma: no cover
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


def _fit_stage_1(step):  # pragma: no cover
    """Fit model using one set of hyperparameters for cross-validation for stage 1.

    :param step: dict; as one element of the return value of _make_stage_1_plan()
    :return: triple of two integers and one float; number of fold, number of trial, score on validation fold
    """
    model, validate_ix = _fit_model(step)
    return step["fold"], step["trial"], model.score(
        X=step["features"][validate_ix],
        y=step["target"][validate_ix],
        sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None
    )


def _fit_model(step):  # pragma: no cover
    """Fit model using one set of hyperparameters.

    :param step: see docstring of _fit_stage_1() for details
    :return: fitted model and 1D numpy bool array indicating the validation fold
    """
    model = step["model"].set_params(**{step["prefix"] + key: value for key, value in step["parameters"].items()})
    model.fit(**{
        "X": step["features"][step["train_ix"]],
        "y": step["target"][step["train_ix"]],
        step["prefix"] + "sample_weight": step["weights"][step["train_ix"]] if step["weights"] is not None else None
    })
    return model, np.logical_not(step["train_ix"])


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


def _execute_stage_2(settings, features, target, weights):  # pragma: no cover
    """Execute search stage 2 for number of batches.

    :param settings: dict; as return value of process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param weights: see docstring of select_hyperparameters() for details
    :return: dict with the following fields:
        - num_batch_grid: as field 'num_batch_grid' of settings
        - scores: 1D numpy float array; mean scores corresponding to num_batch_grid
        - threshold: best score minus standard deviation for the corresponding parameter combination
        - best_index: integer; index for best score
        - selected_index: integer; index for selected score
    """
    stage_2_plan = _make_stage_2_plan(settings=settings, features=features, target=target, weights=weights)
    stage_2 = _execute_plan(
        plan=stage_2_plan,
        num_jobs=settings["num_jobs"],
        fit_function=_fit_stage_2,
        collect_function=np.vstack
    )
    return _evaluate_stage_2(scores=stage_2, num_batch_grid=settings["num_batch_grid"])


def _make_stage_2_plan(settings, features, target, weights):
    """Create cross-validation plan for stage 2.

    :param settings: dict; as return value of _process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param weights: see docstring of select_hyperparameters() for details
    :return: lists of dictionaries; each element has the following fields:
        - model: as input
        - features: as input
        - target: as input
        - weights: as input
        - num_batch_grid: as input
        - prefix: as input
        - train_ix: 1D numpy boolean array; indicator vector for the training set
        - parameters: dict of hyperparameters to be used for fitting; this function specifies n_iter and random_state
    """
    n_iter = settings["num_batch_grid"][-1]
    states = _sample_random_state(size=settings["splitter"].n_splits, random_state=settings["random_state"])
    train_ix = _make_train_ix(features=features, target=target, splitter=settings["splitter"])
    return [{
        "model": deepcopy(settings["model"]),
        "features": features,
        "target": target,
        "weights": weights,
        "num_batch_grid": settings["num_batch_grid"],
        "prefix": settings["prefix"],
        "train_ix": train_ix[i],
        "parameters": {"n_iter": n_iter, "random_state": states[i]}
    } for i in range(len(train_ix))]


def _fit_stage_2(step):  # pragma: no cover
    """Fit model using one set of hyperparameters for cross-validation for stage 2.

    :param step: dict; as one element of the return value of _make_stage_2_plan()
    :return: 1D numpy float array; scores for different numbers of batches to be evaluated
    """
    model, validate_ix = _fit_model(step)
    if step["prefix"] == "":  # model without transformer can be scored normally
        return model.score(
            X=step["features"][validate_ix],
            y=step["target"][validate_ix],
            sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None,
            n_iter=step["num_batch_grid"]
        )
    features = model["transform"].transform(step["features"][validate_ix])
    # sklearn pipeline does not support passing custom parameters to the score function so break it apart
    return model["model"].score(
        X=features,
        y=step["target"][validate_ix],
        sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None,
        n_iter=step["num_batch_grid"]
    )


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
