"""Functions for working with proset models.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks progress of hyperparameter
selection at log level INFO. The invoking application needs to manage log output.
"""

from copy import deepcopy
from functools import partial
import logging
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_random_state


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


MAX_SEED = 1e6
STAGE_1_PERCENTILE = 90  # part of the rule for selecting penalty weights in stage 1
BRIGHT_COLORS = (
    np.array([1.0, 0.2, 0.2]),
    np.array([0.2, 1.0, 0.2]),
    np.array([0.2, 0.2, 1.0])
)
DARK_COLORS = (
    np.array([0.8, 0.0, 0.0]),
    np.array([0.0, 0.8, 0.0]),
    np.array([0.0, 0.0, 0.8])
)


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
):
    """Select hyperparameters for proset model via cross-validation.

    :param model: an instance of a proset model
    :param features: 2D numpy float array; feature matrix; sparse matrices or infinite/missing values not supported
    :param target: list-like object; target for supervised learning
    :param weights: 1D numpy array of positive floats or None; sample weights used for fitting and scoring; pass None to
        use unit weights
    :param transform: sklearn transformer or None; if not None, the transform is applied as part of the model fit to
        normalize features
    :param lambda_v_range: non-negative float, tuple of two non-negative floats, or None; a single value is used as
        penalty for feature weights in all model fits; if a tuple is given, the first value must be strictly less than
        the second and penalties are sampled from the range uniformly on the log scale; pass None to use (1e-6, 1e-1)
    :param lambda_w_range: as lambda_v_range above but for the prototype weights and default (1e-9, 1e-4)
    :param stage_1_trials: positive integer; number of randomized penalty combinations to try in stage 1 of fitting
    :param num_batch_grid: 1D numpy array of non-negative integers or None; batch numbers to try in stage 2 of fitting;
        pass None to use np.linspace(0, 11)
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
    settings = process_select_settings(
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
    logger.info("Execute search stage 1 for penalty weights")
    stage_1 = _execute_stage_1(settings=settings, features=features, target=target, weights=weights)
    logger.info("Execute search stage 2 for number of batches")
    settings["model"].set_params(**{
        settings["prefix"] + "lambda_v": stage_1["lambda_grid"][stage_1["selected_index"], 0],
        settings["prefix"] + "lambda_w": stage_1["lambda_grid"][stage_1["selected_index"], 1]
    })
    stage_2 = _execute_stage_2(settings=settings, features=features, target=target, weights=weights)
    logger.info("Fit final model with selected parameters")
    settings["model"].set_params(**{
        settings["prefix"] + "n_iter": stage_2["num_batch_grid"][stage_2["selected_index"]],
        settings["prefix"] + "random_state": random_state
    })
    settings["model"].fit(**{"X": features, "y": target, settings["prefix"] + "sample_weight": weights})
    logger.info("Hyperparameter selection complete")
    return {"model": settings["model"], "stage_1": stage_1, "stage_2": stage_2}


def process_select_settings(
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
        - num_batch_grid: as input or default if input is None
        - splitter: an sklearn splitter for num_fold folds; stratified K-fold splitter if model is a classifier,
          ordinary K-fold else
        - num_jobs: as input
        - random_state: an instance of numpy.random.RandomState initialized with input random_state
    """
    if lambda_v_range is None:
        lambda_v_range = (1e-6, 1e-1)
    if lambda_w_range is None:
        lambda_w_range = (1e-9, 1e-4)
    if num_batch_grid is None:
        num_batch_grid = np.arange(0, 11)
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
    splitter = StratifiedKFold if is_classifier(model) else KFold
    random_state = check_random_state(random_state)
    return {
        "model": model,
        "prefix": prefix,
        "lambda_v_range": lambda_v_range,
        "lambda_w_range": lambda_w_range,
        "stage_1_trials": stage_1_trials,
        "num_batch_grid": num_batch_grid,
        "splitter": splitter(n_splits=num_folds, shuffle=True, random_state=random_state),
        "num_jobs": num_jobs,
        "random_state": random_state
    }


def _check_select_settings(lambda_v_range, lambda_w_range, stage_1_trials, num_batch_grid, num_jobs):
    """Check input to select_hyperparameters() for consistency.

    :param lambda_v_range: see docstring of select_hyperparameters() for details
    :param lambda_w_range: see docstring of select_hyperparameters() for details
    :param stage_1_trials: see docstring of select_hyperparameters() for details
    :param num_batch_grid: see docstring of select_hyperparameters() for details
    :param num_jobs: see docstring of select_hyperparameters() for details
    :return: no return value; raises an exception if an issue is found
    """
    if isinstance(lambda_v_range, np.float):
        if lambda_v_range < 0.0:
            raise ValueError("Parameter lambda_v must not be negative if passing a single value.")
    else:
        if len(lambda_v_range) != 2:
            raise ValueError("Parameter lambda_v must have length two if passing a tuple.")
        if lambda_v_range[0] < 0.0 or lambda_v_range[1] < 0.0:
            raise ValueError("Parameter lambda_v must not contain negative values if passing a tuple.")
        if lambda_v_range[0] >= lambda_v_range[1]:
            raise ValueError("Parameter lambda_v must contain strictly increasing values if passing a tuple.")
    if isinstance(lambda_w_range, np.float):
        if lambda_w_range < 0.0:
            raise ValueError("Parameter lambda_w must not be negative if passing a single value.")
    else:
        if len(lambda_w_range) != 2:
            raise ValueError("Parameter lambda_w must have length two if passing a tuple.")
        if lambda_w_range[0] < 0.0 or lambda_w_range[1] < 0.0:
            raise ValueError("Parameter lambda_w must not contain negative values if passing a tuple.")
        if lambda_w_range[0] >= lambda_w_range[1]:
            raise ValueError("Parameter lambda_w must contain strictly increasing values if passing a tuple.")
    if stage_1_trials < 1:
        raise ValueError("Parameter stage_1_trials must be a positive integer.")
    if len(num_batch_grid.shape) != 1:
        raise ValueError("Parameters num_batch_grid must be a 1D array.")
    if np.any(num_batch_grid < 0):
        raise ValueError("Parameters num_batch_grid must not contain negative values.")
    if np.any(np.diff(num_batch_grid) <= 0):
        raise ValueError("Parameters num_batch_grid must contain strictly increasing values.")
    if num_jobs is not None and num_jobs < 2:
        raise ValueError("Parameter num_jobs must be at least 2 if passing an integer.")


def _execute_stage_1(settings, features, target, weights):
    """Execute search stage 1 for penalty weights.

    :param settings: dict; as return value of process_select_settings()
    :param features: see docstring of select_hyperparameters() for details
    :param target: see docstring of select_hyperparameters() for details
    :param weights: see docstring of select_hyperparameters() for details
    :return: dict with the following fields:
        - lambda_grid: 2D numpy float array with two columns; values for lambda tried out
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
    return _evaluate_stage_1(stage=stage_1, parameters=stage_1_parameters)


def _make_stage_1_plan(settings, features, target, weights):
    """Create cross-validation plan for stage 1.

    :param settings: dict; as return value of process_select_settings()
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
        random_state=settings["random_state"]
    )
    lambda_w = _sample_lambda(
        lambda_range=settings["lambda_w_range"],
        trials=settings["stage_1_trials"],
        random_state=settings["random_state"]
    )
    states = _sample_random_state(
        size=settings["stage_1_trials"] * settings["splitter"].n_splits,  random_state=settings["random_state"]
    )
    parameters = [{"lambda_v": lambda_v[i], "lambda_w": lambda_w[i]} for i in range(settings["stage_1_trials"])]
    train_ix = _make_train_ix(features=features, target=target, splitter=settings["splitter"])
    return [{
        "model": settings["model"],
        "features": features,
        "target": target,
        "weights": weights,
        "prefix": settings["prefix"],
        "fold": i,
        "train_ix": train_ix[i],
        "trial": j,
        "parameters": {**parameters[j], "random_state": states[i * settings["stage_1_trials"] + j]}
    } for i in range(settings["splitter"].n_splits) for j in range(settings["stage_1_trials"])], parameters


def _sample_lambda(lambda_range, trials, random_state):
    """Sample penalty weights for cross-validation.

    :param lambda_range: see parameter lambda_v_range of select_hyperparameters()
    :param trials: see parameter stage_1_trials of select_hyperparameters()
    :param random_state: an instance of np.random.RandomState
    :return: 1D numpy float array of length trials; penalty weights for cross_validation
    """
    if isinstance(lambda_range, np.float):
        return lambda_range * np.ones(trials)
    offset = random_state.uniform(size=trials)
    logs = np.log(lambda_range)
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
    for i in range(len(folds)):
        new_ix = np.zeros(features.shape[0], dtype=bool)
        new_ix[folds[i][0]] = True
        train_ix.append(new_ix)
    return train_ix


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
        X=step["features"][validate_ix],
        y=step["target"][validate_ix],
        sample_weight=step["weights"][validate_ix] if step["weights"] is not None else None
    )


def _fit_model(step):
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
    for i in range(len(results)):
        scores[results[i][0], results[i][1]] = results[i][2]
    return scores


def _evaluate_stage_1(stage, parameters):
    """Select hyperparameters for stage 1.

    :param stage: as return value of _execute_plan()
    :param parameters: as second return value of _make_stage_1_plan()
    :return: as return value of _execute_stage_1()
    """
    stats = _compute_stats(stage)
    lambda_grid = np.array([[p["lambda_v"], p["lambda_w"]] for p in parameters])
    log_geo_mean = np.sum(np.log(lambda_grid), axis=1)
    candidates = np.logical_and(stats["mean"] >= stats["threshold"], log_geo_mean >= log_geo_mean[stats["best_index"]])
    # candidate points are all points where (a) the mean score is within one standard error of the optimum (equivalent)
    # and (b) the geometric mean of the two penalties is greater or equal to the value at the optimum (more sparse)
    bound = np.percentile(log_geo_mean[candidates], STAGE_1_PERCENTILE)
    candidates = np.logical_and(candidates, log_geo_mean <= bound)
    # remove the candidates that are furthest away from the optimum as there is a risk of making the model too sparse
    selected_index = np.where(log_geo_mean == np.max(log_geo_mean[candidates]))[0][0]
    return {
        "lambda_grid": lambda_grid,
        "scores": stats["mean"],
        "threshold": stats["threshold"],
        "best_index": stats["best_index"],
        "selected_index": selected_index
    }


def _compute_stats(stage):
    """Determine optimal parameter combination and threshold from cross-validation scores.

    :param stage: as return value of _execute_plan()
    :return: dict with the following fields:
        - mean: 1D numpy float array; column means of stage
        - best_index: non-negative integer; index of largest mean score
        - threshold: float; largest mean score minus standard deviation for that parameter combination
    """
    mean = np.mean(stage, axis=0)
    best_index = np.argmax(mean)
    return {
        "mean": mean,
        "best_index": best_index,
        "threshold": mean[best_index] - np.std(stage[:, best_index], ddof=1)
        # use variance correction for small sample size
    }


def _execute_stage_2(settings, features, target, weights):
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
    return _evaluate_stage_2(stage=stage_2, num_batch_grid=settings["num_batch_grid"])


def _make_stage_2_plan(settings, features, target, weights):
    """Create cross-validation plan for stage 2.

    :param settings: dict; as return value of process_select_settings()
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
    n_iter = np.max(settings["num_batch_grid"])
    states = _sample_random_state(size=settings["splitter"].n_splits, random_state=settings["random_state"])
    train_ix = _make_train_ix(features=features, target=target, splitter=settings["splitter"])
    return [{
        "model": settings["model"],
        "features": features,
        "target": target,
        "weights": weights,
        "num_batch_grid": settings["num_batch_grid"],
        "prefix": settings["prefix"],
        "train_ix": train_ix[i],
        "parameters": {"n_iter": n_iter, "random_state": states[i]}
    } for i in range(len(train_ix))]


def _fit_stage_2(step):
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


def _evaluate_stage_2(stage, num_batch_grid):
    """Select hyperparameters for stage 2.

    :param stage: as return value of _execute_plan()
    :param num_batch_grid: see docstring of select_hyperparameters() for details
    :return: as return value of _execute_stage_2()
    """
    stats = _compute_stats(stage)
    selected_index = np.where(stats["mean"] >= stats["threshold"])[0][0]
    return {
        "num_batch_grid": num_batch_grid,
        "scores": stats["mean"],
        "threshold": stats["threshold"],
        "best_index": stats["best_index"],
        "selected_index": selected_index
    }


def print_hyperparameter_report(result):
    """Print report for hyperparameter selection.

    :param result: as return value of select_hyperparameters()
    :return: no return value; results printed to console
    """
    print("{:8s}   {:8s}   {:8s}   {:8s}".format("stage 1", "lambda_v", "lambda_w", "log-loss"))
    print("{:8s}   {:8.1e}   {:8.1e}   {:8.1e}".format(
        "optimal",
        result["stage_1"]["lambda_grid"][result["stage_1"]["best_index"], 0],
        result["stage_1"]["lambda_grid"][result["stage_1"]["best_index"], 1],
        -result["stage_1"]["scores"][result["stage_1"]["best_index"]]
    ))
    print("{:8s}   {:8.1e}   {:8.1e}   {:8.1e}\n".format(
        "selected",
        result["stage_1"]["lambda_grid"][result["stage_1"]["selected_index"], 0],
        result["stage_1"]["lambda_grid"][result["stage_1"]["selected_index"], 1],
        -result["stage_1"]["scores"][result["stage_1"]["selected_index"]]
    ))
    print("{:8s}   {:8s}   {:8s}   {:8s}".format("stage 2", "batches", "", "log-loss"))
    print("{:8s}   {:8d}   {:8s}   {:8.1e}".format(
        "optimal",
        result["stage_2"]["num_batch_grid"][result["stage_2"]["best_index"]],
        "",
        -result["stage_2"]["scores"][result["stage_2"]["best_index"]]
    ))
    print("{:8s}   {:8d}   {:8s}   {:8.1e}\n".format(
        "selected",
        result["stage_2"]["num_batch_grid"][result["stage_2"]["selected_index"]],
        "",
        -result["stage_2"]["scores"][result["stage_2"]["selected_index"]]
    ))


def plot_select_results(result):
    """Plot scores and selected parameters from select_hyperparameters().

    :param result: dict; as return value of select_hyperparameters()
    :return: no return value; plots generated
    """
    _plot_select_stage_1(result["stage_1"])
    _plot_select_parameters(result)


def _plot_select_stage_1(stage_1):
    """Create a surface plot for the penalty weights chosen via select_hyperparameters().

    :param stage_1: dict; as field "stage_1" from return value of select_hyperparameters()
    :return: no return value; plot generated
    """
    levels = np.hstack([np.linspace(min(stage_1["scores"]), stage_1["threshold"], 10), max(stage_1["scores"])])
    best = np.log10(stage_1["lambda_grid"][stage_1["best_index"]])
    selected = np.log10(stage_1["lambda_grid"][stage_1["selected_index"]])
    plt.figure()
    plt.tricontourf(
        np.log10(stage_1["lambda_grid"][:, 0]),
        np.log10(stage_1["lambda_grid"][:, 1]),
        stage_1["scores"],
        levels=levels
    )
    legend = [
        plt.plot(best[0], best[1], "bx", markersize=8, markeredgewidth=2)[0],
        plt.plot(selected[0], selected[1], "r+", markersize=8, markeredgewidth=2)[0]
    ]
    plt.grid(True)
    plt.legend(legend, ["Maximizer", "Selection"])
    plt.title("Hyperparameter search: penalty weights")
    plt.xlabel("Log10(lambda_v)")
    plt.ylabel("Log10(lambda_w)")


def _plot_select_parameters(result):
    """Create plots for the three parameters chosen via select_hyperparameters().

    :param result: dict; as return value of select_hyperparameters()
    :return: no return value; plots generated
    """
    scores = np.hstack([result["stage_1"]["scores"], result["stage_2"]["scores"]])
    y_range = _make_plot_range(scores)
    plt.figure()
    plt.subplot(131)
    _plot_search_1d(
        grid=result["stage_1"]["lambda_grid"][:, 0],
        scores=result["stage_1"]["scores"],
        threshold=result["stage_1"]["threshold"],
        selected_parameter=result["stage_1"]["lambda_grid"][result["stage_1"]["selected_index"], 0],
        y_range=y_range,
        x_label="Lambda_v",
        title="Stage 1: selected lambda_v = {:0.1e}",
        do_show_legend=True
    )
    plt.xscale("log")
    plt.subplot(132)
    _plot_search_1d(
        grid=result["stage_1"]["lambda_grid"][:, 1],
        scores=result["stage_1"]["scores"],
        threshold=result["stage_1"]["threshold"],
        selected_parameter=result["stage_1"]["lambda_grid"][result["stage_1"]["selected_index"], 1],
        y_range=y_range,
        x_label="Lambda_w",
        title="Stage 1: selected lambda_w = {:0.1e}",
        do_show_legend=False
    )
    plt.xscale("log")
    plt.subplot(133)
    _plot_search_1d(
        grid=result["stage_2"]["num_batch_grid"],
        scores=result["stage_2"]["scores"],
        threshold=result["stage_2"]["threshold"],
        selected_parameter=result["stage_2"]["num_batch_grid"][result["stage_2"]["selected_index"]],
        y_range=y_range,
        x_label="Number of batches",
        title="Stage 2: selected number of batches = {}",
        do_show_legend=False
    )
    plt.suptitle("Hyperparameter search")


def _make_plot_range(values, delta=0.05):
    """Compute plot range from vector.

    :param values: 1D numpy float array; values for one plot axis
    :param delta: non-negative float; fraction of the range to be added on both ends of the interval
    :return: 1D numpy float array with 2 elements; minimum and maximum value for plots
    """
    min_value = np.min(values)
    max_value = np.max(values)
    step = (max_value - min_value) * delta
    return np.array([min_value - step, max_value + step])


def _plot_search_1d(grid, scores, threshold, selected_parameter, y_range, x_label, title, do_show_legend):
    """Plot results from parameter search for one parameter.

    :param grid: 1D numpy float array; parameter grid; does not have to be sorted
    :param scores: 1D numpy float array; scores corresponding to grid
    :param threshold: float; threshold for parameter selection
    :param selected_parameter: float; selected parameter value
    :param y_range: numpy float array with two values; y-axis range
    :param x_label: string; x-axis label
    :param title: string; sub-plot title with '{}' format specifier for selected value
    :param do_show_legend: boolean; whether to include legend
    :return: no return value; plot generated in current axes
    """
    order = np.argsort(grid)
    grid = grid[order]
    scores = scores[order]
    x_range = _make_plot_range(grid, delta=0.0)
    legend = [
        plt.plot(grid, scores, linewidth=2, color="k")[0],
        plt.plot(x_range, np.ones(2) * np.max(scores), linewidth=2, color="b", linestyle="--")[0],
        plt.plot(x_range, np.ones(2) * threshold, linewidth=2, color="b")[0],
        plt.plot(np.ones(2) * selected_parameter, y_range, linewidth=2, color="r")[0]
    ]
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(x_label)
    plt.ylabel("Log-likelihood")
    plt.title(title.format(selected_parameter))
    if do_show_legend:
        plt.legend(legend, ["Mean score", "Best score", "Best score - 1 SE", "Selected parameter"], loc="lower left")


def plot_decision_surface(
        features,
        target,
        model,
        feature_names,
        class_labels,
        model_name,
        use_proba=False,
        steps=100,
        num_features=None,
        plot_index=None,
        fixed_features=None,
        classifier_name=None
):
    """Plot decision surface for a classifier with two features and two or three classes.

    :param features: 2D numpy float array with 2 columns; feature matrix
    :param target: list-like object; target for supervised learning
    :param model: fitted classifier supporting the sklearn interface; can have at most three classes
    :param feature_names: tuple of two strings; axis labels for the plot
    :param class_labels: tuple of class labels; used in the legend
    :param model_name: string; name of the model used in the plot title
    :param use_proba: boolean; if True, base surface on predicted probabilities; if false, base it on class
    :param steps: integer greater than 1; number of pixels in each direction
    :param num_features: integer greater than 2 or None; if not None, the total number of features require by the model;
        parameters num_features, plot_features, and fixed_features must either all be None or have values with
        consistent dimensions
    :param plot_index: 1D numpy integer array with 2 elements or None; if not None, indices of the features for plotting
        w.r.t. the full feature matrix
    :param fixed_features: 1D numpy float array with num_features - 2 elements or None; if not None, fixed value for the
        features that are not used for plotting, in order of appearance in the full feature matrix
    :param classifier_name: string or None; if the model is an sklearn pipeline, this must be the name of the actual
        classifier within the pipeline
    :return: no return value, plot generated
    """
    classes = _check_surface_parameters(
        features=features,
        target=target,
        model=model,
        feature_names=feature_names,
        class_labels=class_labels,
        steps=steps,
        num_features=num_features,
        plot_index=plot_index,
        fixed_features=fixed_features,
        classifier_name=classifier_name
    )
    x_range, y_range, colors = _compute_surface(
        features=features,
        model=model,
        use_proba=use_proba,
        steps=steps,
        num_features=num_features,
        plot_index=plot_index,
        fixed_features=fixed_features,
        classes=classes
    )
    plt.figure()
    plt.imshow(
        X=colors,
        interpolation="lanczos" if use_proba else None,
        origin="lower",
        extent=_compute_extent(x_range=x_range, y_range=y_range, steps=steps)
    )
    target = np.searchsorted(classes, target)  # map arbitrary classes to integers
    legend = []
    for i in range(classes.shape[0]):
        ix = target == i
        legend.append(plt.plot(
            features[ix, 0],
            features[ix, 1],
            linestyle="",
            marker="o",
            markeredgewidth=1,
            markeredgecolor="k",
            markersize=6,
            markerfacecolor=DARK_COLORS[i]
        )[0])
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    if use_proba:
        plt.title("Probability surface for {}".format(model_name))
    else:
        plt.title("Decision surface for {}".format(model_name))
    plt.legend(legend, class_labels, bbox_to_anchor=(1, 1), loc="upper left")


def _check_surface_parameters(
        features,
        target,
        model,
        feature_names,
        class_labels,
        steps,
        num_features,
        plot_index,
        fixed_features,
        classifier_name
):
    """Check parameters of plot_decision_surface() for consistency.

    :param features: see docstring of plot_decision_surface() for details
    :param target: see docstring of plot_decision_surface() for details
    :param model: see docstring of plot_decision_surface() for details
    :param feature_names: see docstring of plot_decision_surface() for details
    :param class_labels: see docstring of plot_decision_surface() for details
    :param steps: see docstring of plot_decision_surface() for details
    :param num_features: see docstring of plot_decision_surface() for details
    :param plot_index: see docstring of plot_decision_surface() for details
    :param fixed_features: see docstring of plot_decision_surface() for details
    :param classifier_name: see docstring of plot_decision_surface() for details
    :return: 1D numpy array; classes distinguished by the model, in order; the function raises a ValueError on invalid
        parameters
    """
    features, target = check_X_y(features, target)
    if features.shape[1] != 2:
        raise ValueError("Parameter features must have exactly 2 columns.")
    if classifier_name is None:
        classes = model.classes_
    else:
        classes = model[classifier_name].classes_
    if classes.shape[0] > 3:
        raise ValueError("The model can have at most three classes.")
    if len(feature_names) != 2:
        raise ValueError("Parameter feature_names must have exactly 2 elements.")
    if len(class_labels) != classes.shape[0]:
        raise ValueError("Parameter class_labels must have exactly one element per class.")
    if steps <= 1:
        raise ValueError("Parameters steps must be greater than 2.")
    if num_features is None:
        if plot_index is not None or fixed_features is not None:
            raise ValueError("If parameter num_features is None, plot_features and fixed_features must also be None.")
    else:
        if plot_index is None or fixed_features is None:
            raise ValueError(
                "If parameter num_features is not None, plot_features and fixed_features must also not be None."
            )
        if len(plot_index) != 2:
            raise ValueError("Parameter plot_features must have length 2.")
        if np.any(plot_index < 0) or np.any(plot_index >= num_features):
            raise ValueError("Parameter plot_features must have values between 0 and num_features - 1.")
        if plot_index[0] == plot_index[1]:
            raise ValueError("Parameter plot_features must contain two different values.")
        if fixed_features.shape[0] != num_features - 2:
            raise ValueError("Parameter fixed_features must have length equal to num_features - 2.")
    return classes


def _compute_surface(features, model, use_proba, steps, num_features, plot_index, fixed_features, classes):
    """Determine plot ranges and grid of colors for plotting.

    :param features: see docstring of plot_decision_surface() for details
    :param model: see docstring of plot_decision_surface() for details
    :param use_proba: see docstring of plot_decision_surface() for details
    :param steps: see docstring of plot_decision_surface() for details
    :param num_features: see docstring of plot_decision_surface() for details
    :param plot_index: see docstring of plot_decision_surface() for details
    :param fixed_features: see docstring of plot_decision_surface() for details
    :param classes: 1D numpy array; classes distinguished by the models, in order
    :return: three return values:
        - 1D numpy float array with 2 elements: x-axis plot range
        - 1D numpy float array with 2 elements: y-axis plot range
        - 3D numpy float array of dimension steps, steps, and 3: colors for image plot as RGB values
    """
    x_range = _make_plot_range(features[:, 0])
    y_range = _make_plot_range(features[:, 1])
    grid = np.vstack([
        np.tile(np.linspace(x_range[0], x_range[1], steps), steps),
        np.repeat(np.linspace(y_range[0], y_range[1], steps), steps)
    ]).transpose()
    if num_features is not None:
        extended = np.zeros((grid.shape[0], num_features))
        extended[:, plot_index] = grid
        fixed_index = np.array([i for i in range(num_features) if i not in plot_index])
        extended[:, fixed_index] = fixed_features
        grid = extended
    if use_proba:
        grid_values = model.predict_proba(grid)
    else:
        prediction = np.searchsorted(classes, model.predict(grid))  # convert predictions to integers
        grid_values = np.zeros((prediction.shape[0], classes.shape[0]))
        grid_values[np.arange(prediction.shape[0]), prediction] = 1.0
    colors = np.outer(grid_values[:, 0], BRIGHT_COLORS[0])
    for i in range(1, classes.shape[0]):
        colors += np.outer(grid_values[:, i], BRIGHT_COLORS[i])
    colors = np.reshape(colors, (steps, steps, 3))
    return x_range, y_range, colors


def _compute_extent(x_range, y_range, steps):
    """Compute extent parameter for matplotlib.pyplot.imshow().

    :param x_range: 1D numpy float array with 2 elements; x-axis range for plotting
    :param y_range: 1D numpy float array with 2 elements; y-axis range for plotting
    :param steps: integer greater than 1; number of pixels in each direction
    :return: 1D numpy float array with 4 elements; left, right, bottom, and top coordinate such that the centers of the
        corner pixels match the x_range and y-range coordinates
    """
    return np.hstack([_compute_extent_axis(x_range, steps), _compute_extent_axis(y_range, steps)])


def _compute_extent_axis(axis_range, steps):
    """Compute extent for matplotlib.pyplot.imshow() along one axis.

    :param axis_range: 1D numpy float array with 2 elements; axis range for plotting
    :param steps: integer greater than 1; number of pixels in each direction
    :return: 1D numpy float array with 2 elements
    """
    delta = (axis_range[1] - axis_range[0]) / (2.0 * (steps - 1))
    # the range is covered by steps - 1 pixels with one half of a pixel overlapping on each side; delta is half the
    # pixel width
    return np.array([axis_range[0] - delta, axis_range[1] + delta])
