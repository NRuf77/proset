"""Functions shared by multiple submodules of proset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np


def check_classifier_target(target):
    """Check whether target for classification is encoded correctly.

    :param target: 1D numpy integer array; classes for classification problem encoded as integers from 0 to K - 1; all
        classes must be present
    :return: 1D numpy integer array of counts for each class in order
    """
    classes, counts = np.unique(target, return_counts=True)
    if not np.array_equal(classes, np.arange(classes.shape[0])):
        raise ValueError("The classes must be encoded as integers from 0 to K - 1 and each class must be present.")
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
