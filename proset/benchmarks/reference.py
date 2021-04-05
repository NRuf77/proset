"""Functions for fitting other machine learning models as reference.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from copy import deepcopy

import numpy as np
from scipy.stats.distributions import randint, uniform
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state
import xgboost as xgb


MAX_SEED = 1000000  # maximum value for sampling random seed


def fit_knn_classifier(features, labels, transform=None, k_grid=None, num_folds=5, random_state=None):
    """Fit k-nearest neighbor classifier using log-loss metric.

    :param features: 2D array-like object; feature matrix
    :param labels: list-like object; classification target
    :param transform: sklearn transformer or None; if not None, the transform is applied as part of the model fit to
        normalize features
    :param k_grid: 1D numpy array of positive integers or None; values of k to try; use None to try the numbers from 1
        to 30
    :param num_folds: integer greater than 1; number of cross-validation folds to use
    :param random_state: an instance of numpy.random.RandomState, integer, or None; used as or to initialize a random
        number generator
    :return: dict with the following fields:
        - 'model': if no transform is given, an instance of an sklearn KNeighborsClassifier, else, an sklearn pipeline
           containing a copy of the transformer and the classifier
        - 'info': dict with information on parameter search
    """
    if k_grid is None:
        k_grid = np.arange(1, 31)
    if transform is None:
        model = KNeighborsClassifier()
        para_name = "n_neighbors"
    else:
        model = Pipeline([
            ("transform", deepcopy(transform)),  # do not modify input argument
            ("model", KNeighborsClassifier())
        ])
        para_name = "model__n_neighbors"
    search = GridSearchCV(
        estimator=model,
        param_grid={para_name: k_grid},
        scoring=make_scorer(score_func=log_loss, greater_is_better=False, needs_proba=True),
        cv=StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state),
        refit=False
    )
    search.fit(X=features, y=labels)
    mean = -1.0 * search.cv_results_["mean_test_score"]
    # make_scorer() flips the sign of the scoring function if greater_is_better is False
    std = search.cv_results_["std_test_score"]
    best_index = np.argmin(mean)
    threshold = mean[best_index] + std[best_index]
    selected_index = np.where(mean <= threshold)[0][-1]
    # use the largest k where the score is within one standard error of the optimum to get maximal smoothing
    model.set_params(**{para_name: k_grid[selected_index]})
    model.fit(X=features, y=labels)
    return {
        "model": model,
        "info": {
            "k_grid": k_grid,
            "scores": mean,
            "threshold": threshold,
            "best_index": best_index,
            "selected_index": selected_index
        }
    }


def fit_xgb_classifier(
        features,
        labels,
        eta=None,
        num_iter=None,
        max_depth=None,
        colsample_range=None,
        subsample_range=None,
        stage_1_trials=100,
        num_folds=5,
        random_state=None
):
    """Fit XGBoost classifier using log-loss metric.

    :param features: 2D array-like object; feature matrix
    :param labels: list-like object; classification target
    :param eta: float in (0.0, 1.0], tuple of two floats in (0.0, 1.0], or None; learning rates to be used for the
        initial parameter search and for deciding on the number of boosting iterations; a single number is used for both
        stages; use None for a default of (1e-1, 1e-2)
    :param num_iter: positive integer, tuple of positive integers, or None; number of boosting iterations to be used for
        the initial parameter search and for deciding on the number of boosting iterations; a single number is used for
        both stages; use None for a default of (100, 1000)
    :param max_depth: non-negative integer or None; maximum tree depth is sampled between 0 and this value minus 1; use
        None for a default of 10
    :param colsample_range: float in (0.0, 1.0], tuple of two floats in (0.0, 1.0], or None; a single float fixes the
        colsample_bylevel parameter to that value; if passing two floats, the first must be strictly less than the
        second and the colsample_bylevel parameter is sampled uniformly from that range; pass None to use 1.0
    :param subsample_range: float in (0.0, 1.0], tuple of two floats in (0.0, 1.0], or None; a single float fixes the
        subsample parameter to that value; if passing two floats, the first must be strictly less than the second and
        the subsample parameter is sampled uniformly from that range; pass None to use 1.0
    :param stage_1_trials: positive integer; number of trials for initial parameter selection
    :param num_folds: integer greater than 1; number of cross-validation folds to use
    :param random_state: an instance of numpy.random.RandomState, integer, or None; used as or to initialize a random
        number generator
    :return:
    """
    settings = _process_xgb_classifier_settings(
        labels=labels,
        eta=eta,
        num_iter=num_iter,
        max_depth=max_depth,
        colsample_range=colsample_range,
        subsample_range=subsample_range,
        stage_1_trials=stage_1_trials,
        num_folds=num_folds,
        random_state=random_state
    )
    stage_1 = _fit_xgb_classifier_stage_1(
        features=features,
        labels=labels,
        settings=settings
    )
    stage_2 = _fit_xgb_classifier_stage_2(
        features=features,
        labels=labels,
        settings=settings,
        stage_1=stage_1
    )
    return {
        "model": _fit_final_xgb_classifier(
            features=features,
            labels=labels,
            settings=settings,
            stage_1=stage_1,
            stage_2=stage_2
        ),
        "stage_1": stage_1,
        "stage_2": stage_2
    }


def _process_xgb_classifier_settings(
        labels,
        eta,
        num_iter,
        max_depth,
        colsample_range,
        subsample_range,
        stage_1_trials,
        num_folds,
        random_state
):
    """Validate and prepare setting for fit_xgb_classifier.

    :param labels: see docstring of fit_xgb_classifier()
    :param eta: see docstring of fit_xgb_classifier()
    :param num_iter: see docstring of fit_xgb_classifier()
    :param max_depth: see docstring of fit_xgb_classifier()
    :param colsample_range: see docstring of fit_xgb_classifier()
    :param subsample_range: see docstring of fit_xgb_classifier()
    :param stage_1_trials: see docstring of fit_xgb_classifier()
    :param num_folds: see docstring of fit_xgb_classifier()
    :param random_state: see docstring of fit_xgb_classifier()
    :return: dict with the following fields:
        - num_classes: integer greater than 1; number of classes
        - eta: tuple of two positive floats
        - num_iter: tuple of two positive integers
        - max_depth: as input or default if input is None
        - colsample_range: as input or default if input is None
        - subsample_range: subsample_range,
        - stage_1_trials: as input
        - splitter: an instance of sklearn.model_selection.StratifiedKFold
        - random_state: an instance of np.random.RandomState
    """
    if eta is None:
        eta = (1e-1, 1e-2)
    elif not isinstance(eta, tuple):
        eta = (eta, eta)
    if num_iter is None:
        num_iter = (100, 1000)
    elif not isinstance(num_iter, tuple):
        num_iter = (num_iter, num_iter)
    if max_depth is None:
        max_depth = 10
    if colsample_range is None:
        colsample_range = 1.0
    if subsample_range is None:
        subsample_range = 1.0
    _check_xgb_classifier_settings(
        eta=eta,
        num_iter=num_iter,
        max_depth=max_depth,
        colsample_range=colsample_range,
        subsample_range=subsample_range,
        stage_1_trials=stage_1_trials
    )
    random_state = check_random_state(random_state)
    return {
        "num_classes": np.unique(labels).shape[0],
        "eta": eta,
        "num_iter": num_iter,
        "max_depth": max_depth,
        "colsample_range": colsample_range,
        "subsample_range": subsample_range,
        "stage_1_trials": stage_1_trials,
        "splitter": StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state),
        "random_state": check_random_state(random_state)
    }


def _check_xgb_classifier_settings(
        eta,
        num_iter,
        max_depth,
        colsample_range,
        subsample_range,
        stage_1_trials
):
    """Check input to fit_xgb_classifier() for consistency.

    :param eta: tuple of two floats in (0.0, 1.0]
    :param num_iter: tuple of two positive integers
    :param max_depth: see docstring of fit_xgb_classifier()
    :param colsample_range: see docstring of fit_xgb_classifier()
    :param subsample_range: see docstring of fit_xgb_classifier()
    :param stage_1_trials: see docstring of fit_xgb_classifier()
    :return: no return value; raises a ValueError if an issue is found
    """
    if not _check_ratios(eta):
        raise ValueError("Parameter eta must be a float in (0.0, 1.0] or a tuple of two such values.")
    if num_iter[0] <= 0 or num_iter[1] <= 0:
        raise ValueError("Parameter num_iter must be a positive integer or a tuple of two such values.")
    if max_depth <= 0:
        raise ValueError("Parameter max_depth must be a positive integer.")
    if not _check_ratios(colsample_range):
        raise ValueError("Parameter colsample_range must be a float in (0.0, 1.0] or a tuple of two such values.")
    if isinstance(colsample_range, tuple) and colsample_range[0] >= colsample_range[1]:
        raise ValueError("If passing a tuple for parameter colsample_range, the values must be strictly increasing.")
    if not _check_ratios(subsample_range):
        raise ValueError("Parameter subsample_range must be a float in (0.0, 1.0] or a tuple of two such values.")
    if isinstance(subsample_range, tuple) and subsample_range[0] >= subsample_range[1]:
        raise ValueError("If passing a tuple for parameter subsample_range, the values must be strictly increasing.")
    if stage_1_trials <= 0:
        raise ValueError("Parameter stage_1_trials must be a positive integer.")


def _check_ratios(ratios):
    """Check whether ratios lie in the range (0.0, 1.0].

    :param ratios: float or tuple of two floats
    :return: True iff the ratio or ratio lie in (0.0, 1.0]
    """
    if isinstance(ratios, tuple):
        return 0.0 < ratios[0] <= 1.0 and 0.0 < ratios[1] <= 1.0
    return 0.0 < ratios <= 1.0


def _fit_xgb_classifier_stage_1(features, labels, settings):
    """Perform stage 1 of model fitting.

    :param features: see docstring of fit_xgb_classifier()
    :param labels: see docstring of fit_xgb_classifier()
    :param settings: as return value of def _process_xgb_classifier_settings()
    :return: dict with the following fields:
        - max_depth_grid: 1D numpy array of positive integers; tested values for max_depth
        - colsample_grid: 1D numpy of floats in (0.0, 1.0]; tested values for colsample_bylevel
        - subsample_grid: 1D numpy of floats in (0.0, 1.0]; tested values for subsample
        - scores: 1D numpy float array; log-loss achieved for the different parameter combinations
        - threshold: float; threshold for candidate selection
        - best_index: non-negative integer; index of parameter combination with optimal log-loss
        - selected_index: non-negative integer; index of selected parameter combination
    """
    fixed_para, search_para = _get_xgb_classifier_stage_1_parameters(settings)
    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(**fixed_para),
        param_distributions=search_para,
        n_iter=settings["stage_1_trials"],
        scoring=make_scorer(score_func=log_loss, greater_is_better=False, needs_proba=True),
        cv=settings["splitter"],
        refit=False,
        random_state=settings["random_state"]
    )
    search.fit(X=features, y=labels)
    return _get_xgb_classifier_stage_1_results(search=search, search_para_names=list(search_para.keys()))


def _get_xgb_classifier_stage_1_parameters(settings):
    """Prepare parameters for stage 1 of model fitting.

    :param
    :param settings: as return value of def _process_xgb_classifier_settings()
    :return: two dicts, the first with fixed parameters used by every CV iteration and the second with distributions
        for sampling parameters randomly
    """
    fixed_para = {
        "n_estimators": settings["num_iter"][0],
        "use_label_encoder": False,
        "learning_rate": settings["eta"][0],
        "random_state": settings["random_state"].randint(MAX_SEED)
    }
    fixed_para.update(_get_objective_info(settings["num_classes"]))
    search_para = {"max_depth": randint(low=0, high=settings["max_depth"])}
    if isinstance(settings["colsample_range"], tuple):
        search_para["colsample_bylevel"] = uniform(
            loc=settings["colsample_range"][0], scale=settings["colsample_range"][1] - settings["colsample_range"][0]
        )
    else:
        fixed_para["colsample_bylevel"] = settings["colsample_range"]
    if isinstance(settings["subsample_range"], tuple):
        search_para["subsample"] = uniform(
            loc=settings["subsample_range"][0], scale=settings["subsample_range"][1] - settings["subsample_range"][0]
        )
    else:
        fixed_para["subsample"] = settings["subsample_range"]
    return fixed_para, search_para


def _get_objective_info(num_classes):
    """Provide information on classifier objective.

    :param num_classes: integer greater than 1; number of classes
    :return: dict with information on objective and evaluation metric for XGBoost
    """
    if num_classes == 2:
        return {"objective": "binary:logistic", "eval_metric": "logloss"}
    return {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": num_classes}


def _get_xgb_classifier_stage_1_results(search, search_para_names):
    """Evaluate cross-validation results to choose hyperparameters in stage 1.

    :param search: an instance of sklearn.model_selection.RandomizedSearchCV fitted to data
    :param search_para_names: list of strings; names of all hyperparameters subject to optimization
    :return: as return value of _fit_xgb_classifier_stage_1()
    """
    scores = -1.0 * search.cv_results_["mean_test_score"]
    # make_scorer() flips the sign of the scoring function if greater_is_better is False
    std = search.cv_results_["std_test_score"]
    best_index = np.argmin(scores)
    threshold = scores[best_index] + std[best_index]
    candidates = scores <= threshold
    candidates, max_depth_grid = _update_candidates(
        candidates=candidates,
        search=search,
        search_para_names=search_para_names,
        para_name="max_depth",
        use_min=True  # use smallest depth for individual learners to avoid overfitting
    )
    candidates, colsample_grid = _update_candidates(
        candidates=candidates,
        search=search,
        search_para_names=search_para_names,
        para_name="colsample_bylevel",
        use_min=True  # use smallest fractions of candidate columns per split to get more diverse base learners
    )
    candidates, subsample_grid = _update_candidates(
        candidates=candidates,
        search=search,
        search_para_names=search_para_names,
        para_name="subsample",
        use_min=True  # use smallest number of samples for fitting individual learners to avoid overfitting
    )
    return {
        "max_depth_grid": max_depth_grid,
        "colsample_grid": colsample_grid,
        "subsample_grid": subsample_grid,
        "scores": scores,
        "threshold": threshold,
        "best_index": best_index,
        "selected_index": np.where(candidates)[0][0]
    }


def _update_candidates(candidates, search, search_para_names, para_name, use_min):
    """Narrow selection of candidates for hyperparameters by fixing one of them to the most parsimonious value.

    :param candidates: 1D numpy boolean array; indicator vector of candidates
    :param search: see docstring of _get_xgb_classifier_stage_1_results() for details
    :param search_para_names: see docstring of _get_xgb_classifier_stage_1_results() for details
    :param para_name: string; name of hyperparameter to fix
    :param use_min: boolean; whether to use the smallest or largest value
    :return: two 1D numpy arrays:
        - boolean array: candidates further limited by the indicated hyperparameter
        - float or integer array: parameter search grid for the indicated hyperparameter
    """
    if para_name in search_para_names:
        para_grid = np.array(search.cv_results_["param_" + para_name])  # convert masked array to normal array
    else:
        para_grid = search.estimator.get_params()[para_name] * np.ones_like(candidates)
    if use_min:
        selected_para = np.min(para_grid[candidates])
    else:
        selected_para = np.max(para_grid[candidates])
    candidates = np.logical_and(candidates, para_grid == selected_para)
    return candidates, para_grid


def _fit_xgb_classifier_stage_2(features, labels, settings, stage_1):
    """Perform stage 2 of model fitting.

    :param features: see docstring of fit_xgb_classifier()
    :param labels: see docstring of fit_xgb_classifier()
    :param settings: as return value of def _process_xgb_classifier_settings()
    :param stage_1: as return value of _fit_xgb_classifier_stage_1()
    :return: dict with the following fields:
        - scores: 1D numpy float array; log-loss achieved for each number of iterations from 1 to the desired maximum
        - threshold: float; threshold for candidate selection
        - best_num_iter: positive integer; optimal number of iterations
        - selected_num_iter: positive integer; selected number of iterations
    """
    stage_2_para = _get_xgb_classifier_stage_2_parameters(features, labels, settings, stage_1)
    search = xgb.cv(**stage_2_para)
    # use non-sklearn interface as this provides access to scores for different numbers of boosting iterations
    return _get_xgb_classifier_stage_2_results(search=search, stage_2_para=stage_2_para)


def _get_xgb_classifier_stage_2_parameters(features, labels, settings, stage_1):
    """Prepare parameters for stage 2 of model fitting.

    :param features: see docstring of fit_xgb_classifier()
    :param labels: see docstring of fit_xgb_classifier()
    :param settings: as return value of def _process_xgb_classifier_settings()
    :param stage_1: as return value of _fit_xgb_classifier_stage_1()
    :return: dict of parameter values for fitting XGBoost classifier
    """
    stage_2_para = {
        "params": {
            "eta": settings["eta"][1],
            "max_depth": stage_1["max_depth_grid"][stage_1["selected_index"]],
            "subsample": stage_1["subsample_grid"][stage_1["selected_index"]],
            "colsample_bylevel": stage_1["colsample_grid"][stage_1["selected_index"]]
        },
        "dtrain": xgb.DMatrix(data=features, label=labels),
        "num_boost_round": settings["num_iter"][1],
        "folds": settings["splitter"],
        "stratified": True,
        "seed": settings["random_state"].randint(MAX_SEED)
    }
    stage_2_para["params"].update(_get_objective_info(settings["num_classes"]))
    return stage_2_para


def _get_xgb_classifier_stage_2_results(search, stage_2_para):
    """Evaluate cross-validation results to choose number of boosting iterations in stage 2.

    :param search: pandas data frame with cross-validation results from XGBoost
    :param stage_2_para: as return value of _get_xgb_classifier_stage_2_parameters()
    :return: as return value of _fit_xgb_classifier_stage_2()
    """
    mean_name = "test-{}-mean".format(stage_2_para["params"]["eval_metric"])
    scores = search[mean_name].values
    best_index = np.argmin(scores)
    threshold = scores[best_index] + search[mean_name.replace("-mean", "-std")][best_index]
    return {
        "scores": scores,
        "threshold": threshold,
        "best_num_iter": best_index + 1,
        "selected_num_iter": np.where(scores <= threshold)[0][0] + 1  # use smallest number of iterations
    }


def _fit_final_xgb_classifier(features, labels, settings, stage_1, stage_2):
    """Fit final model with hyperparameter selected via cross-validation.

    :param settings: as return value of def _process_xgb_classifier_settings()
    :param stage_1: as return value of _fit_xgb_classifier_stage_1()
    :param stage_2: as return value of _fit_xgb_classifier_stage_2()
    :return: fitted XGBoost classifier
    """
    model = xgb.XGBClassifier(**_get_xgb_classifier_final_parameters(
        settings=settings, stage_1=stage_1, stage_2=stage_2
    ))
    model.fit(X=features, y=labels)
    return model


def _get_xgb_classifier_final_parameters(settings, stage_1, stage_2):
    """Prepare parameters for stage 2 of model fitting.

    :param settings: as return value of def _process_xgb_classifier_settings()
    :param stage_1: as return value of _fit_xgb_classifier_stage_1()
    :param stage_2: as return value of _fit_xgb_classifier_stage_2()
    :return: dict of parameter values for fitting XGBoost classifier
    """
    final_para = {
        "n_estimators": stage_2["selected_num_iter"],
        "use_label_encoder": False,
        "max_depth": stage_1["max_depth_grid"][stage_1["selected_index"]],
        "learning_rate": settings["eta"][1],
        "subsample": stage_1["subsample_grid"][stage_1["selected_index"]],
        "colsample_bylevel": stage_1["colsample_grid"][stage_1["selected_index"]],
        "random_state": settings["random_state"].randint(MAX_SEED)
    }
    final_para.update(_get_objective_info(settings["num_classes"]))
    return final_para


def print_xgb_classifier_report(result):
    """Print report for hyperparameter selection with XGBoost classifier.

    :param result: as return value of fit_xgb_classifier()
    :return: no return value; results printed to console
    """
    print("{:9s}   {:9s}   {:9s}   {:9s}   {:9s}".format("stage 1", "max_depth", "colsample", "subsample", "log-loss"))
    print("{:9s}   {:9d}   {:9.2f}   {:9.2f}   {:9.2f}".format(
        "optimal",
        result["stage_1"]["max_depth_grid"][result["stage_1"]["best_index"]],
        result["stage_1"]["colsample_grid"][result["stage_1"]["best_index"]],
        result["stage_1"]["subsample_grid"][result["stage_1"]["best_index"]],
        result["stage_1"]["scores"][result["stage_1"]["best_index"]]
    ))
    print("{:9s}   {:9s}   {:9s}   {:9s}   {:9.2f}".format("threshold", "", "", "", result["stage_1"]["threshold"]))
    print("{:9s}   {:9d}   {:9.2f}   {:9.2f}   {:9.2f}".format(
        "selected",
        result["stage_1"]["max_depth_grid"][result["stage_1"]["selected_index"]],
        result["stage_1"]["colsample_grid"][result["stage_1"]["selected_index"]],
        result["stage_1"]["subsample_grid"][result["stage_1"]["selected_index"]],
        result["stage_1"]["scores"][result["stage_1"]["selected_index"]]
    ))
    print("{:9s}   {:9s}   {:9s}   {:9s}   {:9s}".format("stage 2", "num_iter", "", "", "log-loss"))
    print("{:9s}   {:9d}   {:9s}   {:9s}   {:9.2f}".format(
        "optimal",
        result["stage_2"]["best_num_iter"],
        "",
        "",
        result["stage_2"]["scores"][result["stage_2"]["best_num_iter"] - 1]
    ))
    print("{:9s}   {:9s}   {:9s}   {:9s}   {:9.2f}".format("threshold", "", "", "", result["stage_2"]["threshold"]))
    print("{:9s}   {:9d}   {:9s}   {:9s}   {:9.2f}".format(
        "selected",
        result["stage_2"]["selected_num_iter"],
        "",
        "",
        result["stage_2"]["scores"][result["stage_2"]["selected_num_iter"] - 1]
    ))
