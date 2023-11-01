"""Unit tests for code in proset.objectives.shared_classifier.

This module also defines some objects which are not used by the tests in the module but are required by both
test_np_objective.py and test_tf_objective.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

from proset.objectives.np_classifier_objective import NpClassifierObjective
import proset.objectives.shared_classifier as shared_classifier
import proset.shared as shared


# define common objects for testing
FEATURES = np.array([
    [1.0, 0.0, 0.0, -3.7],
    [1.0, 0.0, 0.0, 2.5],
    [0.0, 1.0, 0.0, 4.2],
    [1.0, 0.0, 0.0, -0.8],
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 1.0, 0.0, -2.4],
    [1.0, 0.0, 0.0, 0.7],
    [1.0, 0.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 1.5],
    [0.0, 1.0, 0.0, -0.3]
], **shared.FLOAT_TYPE)
CANDIDATES = np.zeros(FEATURES.shape[0], dtype=bool)
CANDIDATES[:2] = True
REFERENCE = np.logical_not(CANDIDATES)
TARGET = np.array([1, 2, 0, 0, 2, 2, 0, 1, 0, 1])
_, COUNTS = np.unique(TARGET, return_counts=True)
COUNTS_BELOW_FIVE = [str(i) for i in range(len(COUNTS)) if COUNTS[i] < 5]
WEIGHTS = np.linspace(0.1, 2, TARGET.shape[0]).astype(**shared.FLOAT_TYPE)
UNSCALED = np.array([
    [1.0, 11.0, 5.0],  # predicts class 1 correctly
    [4.0, 5.0, 6.0],  # class 2
    [15.0, 12.0, 1.0],  # class 0
    [8.0, 6.0, 7.0],  # class 0
    [1.0, 2.0, 4.0],  # class 2
    [5.0, 3.0, 4.0],  # class 0 instead of 2
    [2.0, 3.0, 2.0],  # class 1 instead of 0
    [6.0, 5.0, 2.0],  # class 0 instead of 1
    [1.0, 4.0, 8.0],  # class 2 instead of 0
    [1.0, 2.0, 3.0]   # class 2 instead of 1
], **shared.FLOAT_TYPE)
SCALE = np.sum(UNSCALED, axis=1)
SCALED = (UNSCALED.transpose() / SCALE).transpose()
GROUPS = np.array([2, 4, 0, 0, 4, 5, 0, 2, 1, 3])
# the sample with the worst prediction per class is selected to be in the corresponding odd-numbered group
LARGE_GROUPS = np.hstack([
    np.zeros(1000, dtype=int),
    np.ones(100, dtype=int),
    2 * np.ones(1000, dtype=int),
    3 * np.ones(200, dtype=int),
])
NUM_LARGE_GROUPS = 4
SAMPLES_PER_LARGE_GROUP = np.array([205, 30, 205, 60])
# based on num_candidates_2 = 500 and max_fraction_2 = 0.3
NUM_CANDIDATES = 1000
NUM_CANDIDATES_2 = 500
MAX_FRACTION = 0.5
MAX_FRACTION_2 = 0.3
MAX_FRACTION_AT_LEAST_FIVE = 0.8
LAMBDA_V = 1e-2
LAMBDA_W = 1e-5
ALPHA_V = 0.95
ALPHA_W = 0.05
BETA = 0.1
RANDOM_STATE = np.random.RandomState(12345)


# pylint: disable=too-few-public-methods, unused-argument
class MockSetManager:
    """Mock SetManager class for interface tests.
    """

    # noinspection PyUnusedLocal
    @staticmethod
    def evaluate_unscaled(features, num_batches):
        """Return constants for interface test.

        :param features: not used
        :param num_batches: not used
        :return: as return value of proset.SetManager.evaluate_unscaled()
        """
        return [(UNSCALED, SCALE)]


# pylint: disable=protected-access
def _get_consistent_example():
    """Create a consistent sample, parameters, and pre-computed results suitable for testing.

    :return: dict with the following keys:
        - sample_data: dict, as output of NpClassifierObjective._split_samples()
        - feature_weights: 1D numpy array of non-negative floats; feature weights vector
        - prototype_weights: 1D numpy array of non-negative floats; prototype weights vector
        - impact: 2D numpy float array; as first return value of NpClassifierObjective._compute_impact()
        - similarity: 2D numpy float array; as second return value of NpClassifierObjective._compute_impact()
    """
    sample_data = NpClassifierObjective._split_samples(  # noqa
        features=FEATURES,
        target=TARGET,
        weights=WEIGHTS,
        beta=BETA,
        num_candidates=NUM_CANDIDATES,
        max_fraction=MAX_FRACTION,
        set_manager=MockSetManager(),
        random_state=RANDOM_STATE,
        meta={"num_classes": COUNTS.shape[0]}
    )
    feature_weights = np.linspace(0.0, 1.0, sample_data["cand_features"].shape[1]).astype(**shared.FLOAT_TYPE)
    prototype_weights = np.linspace(1.0, 2.0, sample_data["cand_features"].shape[0]).astype(**shared.FLOAT_TYPE)
    similarity = NpClassifierObjective._compute_similarity(  # noqa
        feature_weights=feature_weights,
        sample_data=sample_data
    )
    impact = similarity * prototype_weights
    ref_unscaled = sample_data["ref_unscaled"].copy()
    for i in range(ref_unscaled.shape[1]):  # naive implementation as reference: loop over classes
        ix = sample_data["cand_target"] == i
        ref_unscaled[:, i] += np.sum(impact[:, ix], axis=1)
    ref_scale = np.sum(ref_unscaled, axis=1)
    ref_unscaled = np.array([
        ref_unscaled[i, sample_data["ref_target"][i]] for i in range(sample_data["ref_target"].shape[0])
    ])
    return {
        "sample_data": sample_data,
        "feature_weights": feature_weights,
        "prototype_weights": prototype_weights,
        "impact": impact,
        "similarity": similarity,
        "total_weight": np.sum(sample_data["ref_weights"]),
        "ref_unscaled": ref_unscaled,
        "ref_scale": ref_scale
    }


# pylint: disable=missing-function-docstring, protected-access
class TestSharedClassifier(TestCase):
    """Unit tests for code in proset.objectives.shared_classifier.
    """

    def test_check_classifier_init_values_fail_1(self):
        message = ""
        try:
            # test only one exception raised by shared.check_classifier_target() to ensure it is called; other
            # exceptions tested by the unit tests for that function
            shared_classifier.check_classifier_init_values(target=TARGET.astype(float), max_fraction=MAX_FRACTION)
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must be an integer array.")

    def test_check_classifier_init_values_fail_2(self):
        message = ""
        try:
            shared_classifier.check_classifier_init_values(target=TARGET, max_fraction=MAX_FRACTION_AT_LEAST_FIVE)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "\n".join([
            "For the given value of max_fraction, there have to be at least 5 samples per class.",
            "The following classes have fewer cases: {}.".format(", ".join(COUNTS_BELOW_FIVE))
        ]))

    @staticmethod
    def test_check_classifier_init_values_1():
        num_classes = shared_classifier.check_classifier_init_values(target=TARGET, max_fraction=MAX_FRACTION)
        np.testing.assert_array_equal(num_classes, COUNTS.shape[0])

    def test_compute_threshold_1(self):
        result = shared_classifier._compute_threshold(0.1)
        self.assertEqual(result, 9)

    def test_compute_threshold_2(self):
        result = shared_classifier._compute_threshold(0.2)
        self.assertEqual(result, 5)

    def test_compute_threshold_3(self):
        result = shared_classifier._compute_threshold(0.5)
        self.assertEqual(result, 3)

    def test_compute_threshold_4(self):
        result = shared_classifier._compute_threshold(0.8)
        self.assertEqual(result, 5)

    def test_compute_threshold_5(self):
        result = shared_classifier._compute_threshold(0.9)
        self.assertEqual(result, 11)

    def test_assign_groups_1(self):
        num_groups, groups = shared_classifier.assign_groups(
            target=TARGET, beta=BETA, scaled=SCALED, meta={"num_classes": COUNTS.shape[0]}
        )
        self.assertEqual(num_groups, 6)
        np.testing.assert_allclose(groups, GROUPS)

    def test_assign_groups_2(self):
        target = np.hstack([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
        scaled = np.arange(100, dtype=np.float32) / 100.0
        scaled = np.vstack([
            np.hstack([scaled, np.zeros(100, dtype=np.float32)]),
            np.hstack([np.zeros(100, dtype=np.float32), scaled])
        ]).transpose()
        reference = np.hstack([np.ones(25), np.zeros(75), 3 * np.ones(25), 2 * np.ones(75)])
        num_groups, groups = shared_classifier.assign_groups(
            target=target, beta=0.25, scaled=scaled, meta={"num_classes": 2}
        )
        self.assertEqual(num_groups, 4)
        np.testing.assert_allclose(groups, reference)

    def test_find_class_matches_1(self):
        sample_data = {  # include only fields used by the function to be tested
            "ref_features": FEATURES[REFERENCE, :],
            "ref_target": TARGET[REFERENCE],
            "cand_features": FEATURES[CANDIDATES, :],
            "cand_target": TARGET[CANDIDATES]
        }
        class_matches = shared_classifier.find_class_matches(
            sample_data=sample_data,
            meta={"num_classes": COUNTS.shape[0]}
        )
        self.assertTrue(class_matches.flags["F_CONTIGUOUS"])
        class_matches_reference = np.zeros(
            (sample_data["ref_features"].shape[0], sample_data["cand_features"].shape[0]), dtype=bool
        )
        for i in range(sample_data["ref_features"].shape[0]):
            for j in range(sample_data["cand_features"].shape[0]):
                if sample_data["ref_target"][i] == sample_data["cand_target"][j]:
                    class_matches_reference[i, j] = True
        np.testing.assert_array_equal(class_matches, class_matches_reference)
