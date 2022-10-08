"""Various helper functions for working with proset models.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from copy import deepcopy

import numpy as np
from sklearn.base import is_classifier
from sklearn.metrics import pairwise_distances

from proset.shared import check_feature_names, check_scale_offset, LOG_OFFSET


def print_hyperparameter_report(result):  # pragma: no cover
    """Print report for hyperparameter selection.

    :param result: as return value of select_hyperparameters()
    :return: no return value; results printed to console
    """
    cv_results = result["search"]["cv_results"]
    best_ix = result["search"]["best_ix"]
    selected_ix = result["search"]["selected_ix"]
    threshold = -result["search"]["threshold"]
    print("{:9s}   {:>8s}   {:>8s}   {:>8s}".format("solution", "lambda_v", "batches", "log-loss"))
    print("{:9s}   {:8.1e}   {:8d}   {:8.2f}".format(
        "optimal",
        cv_results["lambda_v"].iloc[best_ix],
        cv_results["num_batches"].iloc[best_ix],
        -cv_results["mean_score"].iloc[best_ix]
    ))
    print("{:9s}   {:>8s}   {:>8s}   {:8.2f}".format("threshold", "", "", threshold))
    print("{:9s}   {:8.1e}   {:8d}   {:8.2f}\n".format(
        "selected",
        cv_results["lambda_v"].iloc[selected_ix],
        cv_results["num_batches"].iloc[selected_ix],
        -cv_results["mean_score"].iloc[selected_ix]
    ))


def print_feature_report(model, feature_names=None):  # pragma: no cover
    """Print summary of selected features per batch with weights.

    :param model: a fitted proset model
    :param feature_names: list of strings or None; feature names; pass None to use X0, X1, etc.
    :return: no return value; results printed to console
    """
    report = model.set_manager_.get_feature_weights()
    feature_names = check_feature_names(
        num_features=model.n_features_in_,
        feature_names=feature_names,
        active_features=report["feature_index"]
    )
    max_length = max(np.max([len(name) for name in feature_names]), len("feature"))
    base_format = "{:" + str(max_length) + "s}   "
    header = base_format.format("feature") + "   ".join([
        "{:>8s}".format("batch " + str(i + 1)) for i in range(report["weight_matrix"].shape[0])
    ])
    line_format = base_format + "   ".join(["{:8.2f}"] * report["weight_matrix"].shape[0])
    print(header)
    for i in range(report["weight_matrix"].shape[1]):
        print(line_format.format(feature_names[i], *report["weight_matrix"][:, i]))
    print("")


def choose_reference_point(features, model, scale=None, offset=None):
    """Choose a 'typical' sample as reference point for model explanation.

    :param features: 2D numpy float array; feature matrix
    :param model: fitted proset model
    :param scale: 1D numpy array of positive floats or None; scale for transforming prototype features back to their
        original values; pass None for no transform
    :param offset: 1D numpy array of floats or None; offset for transforming prototype features back to their original
        values; pass None for no transform
    :return: dict with the following fields:
        - index: non-negative integer; row index for selected point in features
        - features_raw: the corresponding row of features as a 2D array
        - features_processed: as above but reduced to active feature and transformed back to original values if scale or
          offset are given
        - prediction: for a classifier, the predicted probabilities for each class belonging to the selected point; for
          a regressor, the predicted target value
        - num_features: positive integer; original number of features
        - active_features: 1D numpy array of non-negative integers; active features for the model
    """
    scale, offset = check_scale_offset(num_features=features.shape[1], scale=scale, offset=offset)
    num_features = features.shape[1]
    if is_classifier(model):
        prediction = model.predict_proba(features)
    else:  # pragma: no cover
        raise NotImplementedError("Function choose_reference_point() does not handle regressors yet.")
    active_features = model.set_manager_.get_active_features()
    index = _find_best_point(
        features=features[:, active_features],
        prediction=prediction,
        is_classifier_=is_classifier(model)
    )
    return {
        "index": index,
        "features_raw": features[index:(index + 1), :].copy(),
        "features_processed":
            features[index:(index + 1), active_features] * scale[active_features] + offset[active_features],
        "prediction": prediction[index],
        "num_features": num_features,
        "active_features": active_features
    }


def _find_best_point(features, prediction, is_classifier_):
    """Identify the best reference point for model explanation.

    :param features: 2D numpy float array; feature matrix
    :param prediction: numpy array; to explain a classifier, pass the matrix of predicted probabilities corresponding to
        the features; regressors are not supported yet
    :param is_classifier_: boolean; whether the model to be explained is a classifier or regressor
    :return: non-negative integer; row index
    """
    if not is_classifier_:  # pragma: no cover
        raise NotImplementedError("Function choose_reference_point() does not handle regressors yet.")
    points = _compute_borda_points(np.mean(pairwise_distances(prediction), axis=1))
    if features.shape[1] > 0:  # safeguard in case the model has no active features
        points += _compute_borda_points(np.mean(pairwise_distances(features), axis=1))
    candidates = np.nonzero(points == np.max(points))[0]
    if candidates.shape[0] > 1:
        entropy = -np.sum(np.log(prediction[candidates] * LOG_OFFSET) * prediction[candidates], axis=1)
        # this formulation is for classifiers only
        return candidates[np.argmax(entropy)]
    return candidates[0]


def _compute_borda_points(metric):
    """Compute points for Borda voting with ties from value of a metric that needs to be minimized.

    Each sample is assigned a full point for every other sample that has a strictly larger value of the metric and a
    half point for each sample with the same value.

    :param metric: 1D numpy float array; metric values - lower is better
    :return: 1D numpy float array of the same length as metric; points assigned to each score
    """
    inverse, counts = np.unique(metric, return_inverse=True, return_counts=True)[1:]
    points = (counts - 1) / 2.0 + np.hstack([0, np.cumsum(counts[-1:0:-1])])[-1::-1]
    return points[inverse]


def print_point_report(reference, feature_names=None, feature_format="1f", target_names=None):  # pragma: no cover
    """Print information on selected point.

    :param reference: as return value of choose_reference_point()
    :param feature_names: list of strings or None; feature names; pass None to use X0, X1, etc.
    :param feature_format: string; format specifier for feature values converted to string; provide only decimals and
        convention ('f' for float, 'e' for scientific, etc.)
    :param target_names: string, list of strings, or None; for a regression problem, pass a single string or None to use
        no name; for a classification problem, pass a list of class labels or None to use integer labels
    :return: no return value; results printed to console
    """
    feature_names, target_names, is_classifier_ = _check_point_input(
        reference=reference,
        feature_names=feature_names,
        target_names=target_names
    )
    print("Properties of point with sample index {}:".format(reference["index"]))
    max_length = np.max([len(name) for name in feature_names] + [len("feature")])
    base_format = "{:>" + str(max_length) + "s}  "
    header_format = base_format + "{:>8s}"
    row_format = base_format + "{:8." + feature_format + "}"
    print(header_format.format("feature", "value"))
    for i, name in enumerate(feature_names):
        print(row_format.format(name, reference["features_processed"][0, i]))
    print("Prediction for point with sample index {}:".format(reference["index"]))
    if is_classifier_:
        max_length = np.max([len(name) for name in target_names] + [len("class")])
        base_format = "{:>" + str(max_length) + "s}  "
        header_format = base_format + "{:>10s}"
        row_format = base_format + "{:11.2f}"
        print(header_format.format("class", "probability"))
        for i, name in enumerate(target_names):
            print(row_format.format(name, reference["prediction"][i]))
    else:
        print("The predicted target value {} is {}.".format(target_names, reference["prediction"]))


def _check_point_input(reference, feature_names, target_names):
    """Check input for print_point_report() for consistency and apply defaults where required.

    :param reference: see docstring of print_point_report() for details
    :param feature_names: see docstring of print_point_report() for details
    :param target_names: see docstring of print_point_report() for details
    :return: three return values:
        - list of strings: feature names; feature names for active features
        - string or list of strings: target names; as input or defaults if input is None
        - boolean: True if reference belongs to a classifier, False if it belongs to a regressor
    """
    feature_names = check_feature_names(
        num_features=reference["num_features"],
        feature_names=feature_names,
        active_features=reference["active_features"]
    )
    is_classifier_ = isinstance(reference["prediction"], np.ndarray)
    if is_classifier_:
        if isinstance(target_names, str):
            raise TypeError(
                "Parameter target_names must be a list of strings or None if reference belongs to a classifier."
            )
        num_classes = reference["prediction"].shape[0]
        if target_names is not None:
            if len(target_names) != num_classes:
                raise ValueError(" ".join([
                    "Parameter target_names must have one element per class if passing a list",
                    "and reference belongs to a classifier."
                ]))
            target_names = deepcopy(target_names)
        else:
            target_names = [str(i) for i in range(num_classes)]
    else:
        if isinstance(target_names, list):
            raise TypeError("Parameter target_names must be a string or None if reference belongs to a regressor.")
        if target_names is None:
            target_names = "value"
    return feature_names, target_names, is_classifier_
