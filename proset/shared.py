"""Functions shared by multiple submodules of proset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np
import scipy.linalg.blas as blas  # pylint: disable=no-name-in-module


FLOAT_TYPE = {"dtype": np.float32, "order": "F"}  # enforce this format for all float arrays
LOG_OFFSET = 1e-10  # add to small numbers before taking the logarithm


def check_classifier_target(target, weights):
    """Check whether target for classification is encoded correctly.

    :param target: 1D numpy integer array; classes for classification problem encoded as integers from 0 to K - 1; all
        classes must be present
    :param weights: 1D numpy array with non-negative values of type specified by FLOAT_TYPE or None; sample weights to
        be used in the likelihood function; pass None to use unit weights
    :return: 1D numpy array with non-negative values of type specified by FLOAT_TYPE; counts for each class in order
    """
    if len(target.shape) != 1:
        raise ValueError("Parameter target must be a 1D array.")
    if not np.issubdtype(target.dtype, np.integer):
        raise TypeError("Parameter target must be an integer array.")
    if weights is None:
        classes, counts = np.unique(target, return_counts=True)
        counts = counts.astype(**FLOAT_TYPE)
    else:
        if len(weights.shape) != 1:
            raise ValueError("Parameter weights must be a 1D array.")
        if weights.shape[0] != target.shape[0]:
            raise ValueError("Parameter weights must have as many elements as target.")
        if np.any(weights < 0.0):
            raise ValueError("Parameter weights must not contain negative values.")
        check_float_array(x=weights, name="weights")
        classes = np.unique(target)
        sort_ix = np.argsort(target)
        # np.add.reduceat() requires weights with the same target value to be grouped together
        changes = find_changes(target[sort_ix])
        counts = np.add.reduceat(weights[sort_ix], indices=changes)
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

    :param scaled_reference: 2D numpy array of type specified by shared.FLOAT_TYPE; features for reference points scaled
        with feature weights
    :param scaled_prototypes: 2D numpy array of type specified by shared.FLOAT_TYPE; features for prototypes scaled with
        feature weights; must have as many columns as scaled_reference
    :param ssq_reference: 1D numpy array of type specified by shared.FLOAT_TYPE; the row-sums of scaled_reference after
        squaring the values
    :param ssq_prototypes: 1D numpy array of type specified by shared.FLOAT_TYPE; the row-sums of scaled_prototypes
        after squaring the values
    :return: 2D array with positive values of type specified by shared.FLOAT_TYPE with one row per sample and one column
        per prototype
    """
    similarity = -2.0 * blas.sgemm(  # pylint: disable=no-member
        alpha=1.0, a=scaled_reference, b=scaled_prototypes, trans_b=1
    )
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
        - 1D numpy array with positive values of type specified by FLOAT_TYPE; as scale if given, else a vector of ones
        - 1D numpy array of type specified by FLOAT_TYPE; as offset if given, else a vector of zeros
    """
    if not np.issubdtype(type(num_features), np.integer):
        raise TypeError("Parameter num_features must be integer.")
    if num_features < 1:
        raise ValueError("Parameter num_features must be positive.")
    if scale is None:
        scale = np.ones(num_features, **FLOAT_TYPE)
    else:
        if len(scale.shape) != 1:
            raise ValueError("Parameter scale must be a 1D array.")
        if scale.shape[0] != num_features:
            raise ValueError("Parameter scale must have one element per feature.")
        if np.any(scale <= 0.0):
            raise ValueError("Parameter scale must have strictly positive elements.")
        scale = scale.astype(**FLOAT_TYPE)  # this implicitly makes a copy so original input is not affected
    if offset is None:
        offset = np.zeros(num_features, **FLOAT_TYPE)
    else:
        if len(offset.shape) != 1:
            raise ValueError("Parameter offset must be a 1D array.")
        if offset.shape[0] != num_features:
            raise ValueError("Parameter offset must have one element per feature.")
        offset = offset.astype(**FLOAT_TYPE)
    return scale, offset


def check_float_array(x, name, spec=None):
    """Check that a float array has the required data type and order.

    :param x: numpy array
    :param name: parameter name to show in exception messages
    :param spec: dict or None; pass None to use module-level property FLOAT_TYPE as default; if not None, must have the
        same format as FLOAT_TYPE
    :return: no return value, raises an exception if the check fails
    """
    if spec is None:
        spec = FLOAT_TYPE
    if x.dtype != spec["dtype"]:
        raise TypeError("Parameter {} must be an array of type {}.".format(name, spec["dtype"].__name__))
    if not x.flags["F_CONTIGUOUS"] and spec["order"] == "F":
        raise TypeError("Parameter {} must be a Fortran-contiguous array.".format(name))
    if not x.flags["C_CONTIGUOUS"] and spec["order"] == "C":
        raise TypeError("Parameter {} must be a C-contiguous array.".format(name))


def stack_first(array_list):
    """Stack list of 1D or 2D numpy arrays along first dimension.

    :param array_list: list of numpy arrays, all either 1D or 2D; 2D arrays must have the same number of columns each
    :return: numpy array; input list stacked along first dimension (axis 0)
    """
    if len(array_list[0].shape) == 1:
        return np.hstack(array_list)
    return np.vstack(array_list)
