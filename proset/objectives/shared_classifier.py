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


def assign_groups(target, beta, scaled, meta):
    """Divide training samples into groups for sampling candidates.

    :param target: see docstring of objective.NpClassifierObjective._check_init_values() for details
    :param beta: see docstring of objective.Objective.__init__() for details
    :param scaled: see docstring of objective.Objective._assign_groups() for details
    :param meta: dict; must have key 'num_classes' referencing the number of classes
    :return: as return value of objective.Objective._assign_groups()
    """
    goodness = np.squeeze(np.take_along_axis(scaled, target[:, None], axis=1))
    # extract the predicted probability for the true class
    groups = np.zeros_like(target, dtype=int)
    for class_ in range(meta["num_classes"]):
        ix = target == class_
        groups[ix] = 2 * class_ + (goodness[ix] < np.quantile(a=goodness[ix], q=beta)).astype(int)
        # assign observations with low goodness to odd groups
    return 2 * meta["num_classes"], groups


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
