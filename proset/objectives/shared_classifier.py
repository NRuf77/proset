"""Functions shared by numpy and tensorflow classifier objective functions.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np

import proset.shared as shared


def check_classifier_init_values(target, max_fraction):
    """Perform input validation common to all classifier objectives.

    :param target: 1D numpy integer array; target for classification
    :param max_fraction: see docstring of objective.Objective.__init__() for details
    :return: integer; number of classes
    """
    counts = shared.check_classifier_target(target=target, weights=None)
    # the check whether samples can be split into prototypes and candidates does not consider weights
    threshold = _compute_threshold(max_fraction)
    too_few_cases = counts < threshold
    if np.any(too_few_cases):
        raise ValueError("\n".join([
            "For the given value of max_fraction, there have to be at least {} samples per class.".format(
                threshold
            ),
            "The following classes have fewer cases: {}.".format(", ".join(
                ["{}".format(x) for x in np.nonzero(too_few_cases)[0]]
            ))
        ]))
    return counts.shape[0]


def _compute_threshold(max_fraction):
    """Compute required minimum of samples per class.

    :param max_fraction: see docstring of objective.Objective.__init__() for details
    :return: positive integer
    """
    half_samples = max(np.ceil(0.5 / max_fraction), np.floor(0.5 / (1.0 - max_fraction)) + 1.0)
    return 2 * int(half_samples) - 1


def assign_groups(target, unscaled, meta):
    """Divide training samples into groups for sampling candidates.

    :param target: see docstring of objective.Objective.__init__() for details
    :param unscaled: 2D numpy array of type specified by shared.FLOAT_TYPE; unscaled predictions corresponding to the
        target values
    :param meta: dict; must have key 'num_classes' referencing the number of classes
    :return: two return values:
        - integer; total number of groups mandated by hyperparameters
        - 1D numpy integer array; group assignment to samples as integer from 0 to the number of groups - 1; note
          that the assignment is not guaranteed to contain all group numbers
    """
    groups = 2 * target  # correctly classified samples are assigned an even group number based on their target
    groups[target != np.argmax(unscaled, axis=1)] += 1  # incorrectly classified samples are assigned odd numbers
    # no need to scale as the scales for classification are just the row-sums
    return 2 * meta["num_classes"], groups


def adjust_ref_weights(sample_data, candidates, target, weights, meta):
    """Adjust weights of reference points in sample data for classification.

    :param sample_data: as return value of objective.Objective._finalize_split()
    :param candidates: as return value of objective.Objective._sample_candidates()
    :param target: see docstring of objective.Objective.__init__() for details
    :param weights: see docstring of objective.Objective.__init__() for details
    :param meta: dict; must have key 'num_classes' referencing the number of classes
    :return: 1D numpy array with positive values of type specified by shared.FLOAT_TYPE; adjusted reference weights
    """
    ref_weights = sample_data["ref_weights"].copy(order="F")  # do not modify original input
    for i in range(meta["num_classes"]):
        ix = target == i
        class_weight = np.sum(weights[ix])
        class_weight /= class_weight - np.sum(weights[np.logical_and(ix, candidates)])
        ix = sample_data["ref_target"] == i
        ref_weights[ix] = ref_weights[ix] * class_weight
    return ref_weights


def find_class_matches(sample_data, meta):
    """Determine for each candidate and reference point in sample data whether they have matching classes.

    :param sample_data: as return value of objective.Objective._finalize_split()
    :param meta: dict; must have key 'num_classes' referencing the number of classes
    :return: 2D numpy F-contiguous boolean array with one row per reference point and one column per prototype
        candidate; indicates whether the respective reference and candidate point have the same class
    """
    class_matches = np.zeros(
        (sample_data["ref_features"].shape[0], sample_data["cand_features"].shape[0]), dtype=bool, order="F"
    )
    for i in range(meta["num_classes"]):
        class_matches += np.outer(sample_data["ref_target"] == i, sample_data["cand_target"] == i)
    return class_matches
