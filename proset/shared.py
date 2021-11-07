"""Functions shared by multiple submodules of proset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np


LOG_OFFSET = 1e-10  # add to small numbers before taking the logarithm


def check_classifier_target(target):
    """Check whether target for classification is encoded correctly.

    :param target: 1D numpy integer array; classes for classification problem encoded as integers from 0 to K - 1; all
        classes must be present
    :return: 1D numpy integer array of counts for each class in order
    """
    if len(target.shape) != 1:
        raise ValueError("Parameter target must be a 1D array.")
    if not np.issubdtype(target.dtype, np.integer):
        raise TypeError("Parameter target must be an integer array.")
    classes, counts = np.unique(target, return_counts=True)
    if not np.array_equal(classes, np.arange(classes.shape[0])):
        raise ValueError(
            "Parameter target must encode classes as integers from 0 to K - 1 and every class must be present."
        )
    return counts


def find_changes(x):
    """Return indices of all elements in a vector where the value changes, including the first.

    :param x: 1D numpy array
    :return: 1D numpy array of non-negative integers; indices of changes in order
    """
    return np.hstack([0, np.nonzero(np.diff(x))[0] + 1])


def quick_compute_similarity(scaled_reference, scaled_prototypes, ssq_reference, ssq_prototypes):
    """Compute similarity between prototypes and reference points.

    :param scaled_reference: 2D numpy float array; features for reference points scaled with feature weights
    :param scaled_prototypes: 2D numpy float array; features for prototypes scaled with feature weights; must have as
        many columns as scaled_reference
    :param ssq_reference: 1D numpy float array; the row-sums of scaled_reference after squaring the values
    :param ssq_prototypes: 1D numpy float array; the row-sums of scaled_prototypes after squaring the values
    :return: 2D numpy array of positive floats with one row per sample and one column per prototype
    """
    similarity = -2.0 * np.inner(scaled_reference, scaled_prototypes)
    similarity += ssq_prototypes
    similarity = (similarity.transpose() + ssq_reference).transpose()  # broadcast over columns
    similarity = np.exp(-0.5 * similarity)
    return similarity


def check_feature_names(num_features, feature_names, active_features):
    """Check feature names for consistency and supply defaults if necessary.

    :param num_features: positive integer; number of features
    :param feature_names: list of strings or None; if not None, must have num_features elements
    :param active_features: 1D numpy integer array or None; feature indices to select; defaults to all features in
        the order given in feature_names
    :return: list of strings; feature names or defaults X0, X1, etc.; raise an error on invalid input
    """
    if not np.issubdtype(type(num_features), np.integer):
        raise TypeError("Parameter num_features must be integer.")
    if num_features < 1:
        raise ValueError("Parameter num_features must be positive.")
    if feature_names is not None and len(feature_names) != num_features:
        raise ValueError("Parameter feature_names must have one element per feature if not None.")
    if active_features is None:
        active_features = np.arange(num_features)
    if len(active_features.shape) != 1:
        raise ValueError("Parameter active_features must be a 1D array.")
    if not np.issubdtype(active_features.dtype, np.integer):
        raise TypeError("Parameter active_features must be an integer array.")
    if np.any(active_features < 0) or np.any(active_features >= num_features):
        raise ValueError(
            "Parameter active_features must contain non-negative numbers less than the total number of features."
        )
    if feature_names is None:
        return ["X{}".format(i) for i in active_features]
    return [feature_names[i] for i in active_features]


def check_scale_offset(num_features, scale, offset):
    """Check that scale and offset are consistent with number of features, provide defaults if necessary.

    :param num_features: non-negative integer; number of features
    :param scale: 1D numpy array of positive floats or None; if not None, must have num_features elements
    :param offset: 1D numpy float array or None; if not None, must have num_features elements
    :return: two return values:
        - 1D numpy array of positive floats; as scale if given, else a vector of ones
        - 1D numpy float array; as offset if given, else a vector of zeros
    """
    if not np.issubdtype(type(num_features), np.integer):
        raise TypeError("Parameter num_features must be integer.")
    if num_features < 1:
        raise ValueError("Parameter num_features must be positive.")
    if scale is None:
        scale = np.ones(num_features, dtype=float)
    else:
        if len(scale.shape) != 1:
            raise ValueError("Parameter scale must be a 1D array.")
        if scale.shape[0] != num_features:
            raise ValueError("Parameter scale must have one element per feature.")
        if np.any(scale <= 0.0):
            raise ValueError("Parameter scale must have strictly positive elements.")
        scale = scale.copy()  # output should not be a reference to input
    if offset is None:
        offset = np.zeros(num_features, dtype=float)
    else:
        if len(offset.shape) != 1:
            raise ValueError("Parameter offset must be a 1D array.")
        if offset.shape[0] != num_features:
            raise ValueError("Parameter offset must have one element per feature.")
        offset = offset.copy()
    return scale, offset
