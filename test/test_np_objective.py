"""Unit tests for code in proset.objectives that relies on numpy for evaluation.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

from proset import shared
from proset.objectives.objective import START_FEATURE_WEIGHT, START_PROTOTYPE_WEIGHT
from proset.objectives.np_classifier_objective import NpClassifierObjective
# pylint: disable=wrong-import-order
from test.test_shared_classifier import MockSetManager, _get_consistent_example, FEATURES, TARGET, COUNTS, \
    COUNTS_BELOW_FIVE, WEIGHTS, UNSCALED, SCALE, SCALED, GROUPS, LARGE_GROUPS, NUM_LARGE_GROUPS, \
    SAMPLES_PER_LARGE_GROUP, NUM_CANDIDATES, NUM_CANDIDATES_2, MAX_FRACTION, MAX_FRACTION_2, \
    MAX_FRACTION_AT_LEAST_FIVE, LAMBDA_V, LAMBDA_W, ALPHA_V, ALPHA_W, BETA, RANDOM_STATE


def _choose_half(x):
    """Return an index vector referencing the first half of a vector.

    :param x: 1D numpy array
    :return: 1D numpy integer array; indices referencing the first half of x (excludes the middle in case of odd length)
    """
    return np.nonzero(np.arange(x.shape[0]) < x.shape[0] / 2)[0]


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestNpClassifierObjective(TestCase):
    """Unit tests for class NpClassifierObjective.

    The tests also cover abstract superclasses Objective and NpObjective.
    """

    def test_init_fail_1(self):
        message = ""
        try:  # trigger one check from shared.check_classifier_target() to ensure it is called
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET[:, np.newaxis],
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must be a 1D array.")

    def test_init_fail_2(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION_AT_LEAST_FIVE,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "\n".join([
            "For the given value of max_fraction, there have to be at least 5 samples per class.",
            "The following classes have fewer cases: {}.".format(", ".join(COUNTS_BELOW_FIVE))
        ]))

    def test_init_fail_3(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES[:, 0],
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features must be a 2D array.")

    def test_init_fail_4(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES[0:1, :],
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "The training data needs to contain more than one sample.")

    def test_init_fail_5(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET[:-1],
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must have as many elements as features has rows.")

    def test_init_fail_6(self):
        message = ""
        try:  # trigger one check from shared.check_float_array() to ensure it is called
            NpClassifierObjective(
                features=FEATURES.astype(np.float64),
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features must be an array of type float32.")

    def test_init_fail_7(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS[:, np.newaxis],
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter weights must be a 1D array.")

    def test_init_fail_8(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS[:-1],
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter weights must have as many elements as features has rows.")

    def test_init_fail_9(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=-WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter weights must not contain negative values.")

    def test_init_fail_10(self):
        message = ""
        try:  # trigger one check from shared.check_float_array() to ensure it is called
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS.astype(np.float64),
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter weights must be an array of type float32.")

    def test_init_fail_11(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=0.0,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter beta must lie in (0.0, 0.5].")

    def test_init_fail_12(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=0.6,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter beta must lie in (0.0, 0.5].")

    def test_init_fail_13(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=1.0,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_candidates must be integer.")

    def test_init_fail_14(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=0,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_candidates must be positive.")

    def test_init_fail_15(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=0.0,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_fraction must lie in (0.0, 1.0).")

    def test_init_fail_16(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=1.0,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_fraction must lie in (0.0, 1.0).")

    def test_init_fail_17(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=-1.0,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must not be negative.")

    def test_init_fail_18(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=-1.0,
                alpha_v=ALPHA_V,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_w must not be negative.")

    def test_init_fail_19(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=-1.0,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter alpha_v must lie in [0.0, 1.0].")

    def test_init_fail_20(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=2.0,
                alpha_w=ALPHA_W,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter alpha_v must lie in [0.0, 1.0].")

    def test_init_fail_21(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=-1.0,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter alpha_w must lie in [0.0, 1.0].")

    def test_init_fail_22(self):
        message = ""
        try:
            NpClassifierObjective(
                features=FEATURES,
                target=TARGET,
                weights=WEIGHTS,
                beta=BETA,
                num_candidates=NUM_CANDIDATES,
                max_fraction=MAX_FRACTION,
                set_manager=MockSetManager(),
                lambda_v=LAMBDA_V,
                lambda_w=LAMBDA_W,
                alpha_v=ALPHA_V,
                alpha_w=2.0,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter alpha_w must lie in [0.0, 1.0].")

    def test_init_1(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        self.assertEqual(len(objective._meta), 4)
        np.testing.assert_allclose(objective._meta["num_classes"], COUNTS.shape[0])
        dimensions = objective._sample_data["cand_features"].shape
        self.assertEqual(objective._meta["num_features"], dimensions[1])
        self.assertEqual(objective._meta["num_parameters"], dimensions[1] + dimensions[0])
        self.assertEqual(objective._meta["total_weight"], np.sum(WEIGHTS))
        self.assertEqual(len(objective._sample_data), 13)  # a full check is performed by test_split_samples_1() below
        self.assertEqual(objective._sample_cache, None)
        self.assertEqual(objective._lambda_v, LAMBDA_V)
        self.assertEqual(objective._lambda_w, LAMBDA_W)
        self.assertEqual(objective._alpha_v, ALPHA_V)
        self.assertEqual(objective._alpha_w, ALPHA_W)

    def test_init_2(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=None,  # use unit weights
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )  # test only behavior for unit weights, rest is as above
        self.assertEqual(objective._meta["total_weight"], FEATURES.shape[0])
        shared.check_float_array(x=objective._sample_data["ref_weights"], name="objective._sample_data['ref_weights']")

    # method _check_init_values() already tested by the above

    def test_split_samples_1(self):
        result = NpClassifierObjective._split_samples(
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
        self.assertEqual(len(result), 13)
        shared.check_float_array(x=result["ref_features"], name="result['ref_features']")
        candidates = np.zeros(FEATURES.shape[0], dtype=bool)
        candidates[result["cand_index"]] = True  # need to reconstruct candidates from result due to randomization
        reference = np.logical_not(candidates)
        np.testing.assert_allclose(result["ref_features"], FEATURES[reference])
        shared.check_float_array(x=result["ref_features_squared"], name="result['ref_features_squared']")
        np.testing.assert_allclose(result["ref_features_squared"], FEATURES[reference] ** 2.0)
        ref_target = TARGET[reference]
        np.testing.assert_allclose(result["ref_target"], ref_target)
        shared.check_float_array(x=result["ref_weights"], name="result['ref_weights']")
        group_scales = np.zeros(max(GROUPS) + 1, dtype=float)
        for i in range(group_scales.shape[0]):
            ix = GROUPS == i
            group_scales[i] = np.sum(WEIGHTS[ix]) / np.sum(WEIGHTS[np.logical_and(ix, reference)])
        ref_weights = WEIGHTS[reference]
        for i, target in enumerate(GROUPS[reference]):
            ref_weights[i] *= group_scales[target]
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.sum(ref_weights), np.sum(WEIGHTS), delta=1e-6)
        # check whether reference weights are consistent
        np.testing.assert_allclose(result["ref_weights"], ref_weights, atol=1e-6)
        shared.check_float_array(x=result["ref_unscaled"], name="result['ref_unscaled']")
        np.testing.assert_allclose(result["ref_unscaled"], UNSCALED[reference])
        shared.check_float_array(x=result["ref_scale"], name="result['ref_scale']")
        np.testing.assert_allclose(result["ref_scale"], SCALE[reference])
        order = np.argsort(TARGET[candidates])
        shared.check_float_array(x=result["cand_features"], name="result['cand_features']")
        np.testing.assert_allclose(result["cand_features"], FEATURES[candidates][order])
        shared.check_float_array(x=result["cand_features_squared"], name="result['cand_features_squared']")
        np.testing.assert_allclose(result["cand_features_squared"], FEATURES[candidates][order] ** 2.0)
        np.testing.assert_allclose(result["cand_target"], TARGET[candidates][order])
        np.testing.assert_allclose(result["cand_index"], np.nonzero(candidates)[0][order])
        self.assertEqual(
            result["shrink_sparse_features"],
            ["ref_features", "ref_features_squared", "cand_features", "cand_features_squared"]
        )
        np.testing.assert_allclose(result["cand_changes"], shared.find_changes(TARGET[candidates][order]))
        class_matches = np.zeros((result["ref_features"].shape[0], result["cand_features"].shape[0]), dtype=bool)
        for i in range(result["ref_features"].shape[0]):
            for j in range(result["cand_features"].shape[0]):
                class_matches[i, j] = result["ref_target"][i] == result["cand_target"][j]
        np.testing.assert_allclose(result["class_matches"], class_matches)

    def test_assign_groups_1(self):
        num_groups, groups = NpClassifierObjective._assign_groups(
            target=TARGET, beta=BETA, scaled=SCALED, meta={"num_classes": COUNTS.shape[0]}
        )
        self.assertEqual(num_groups, 6)
        np.testing.assert_allclose(groups, GROUPS)

    def test_sample_candidates_1(self):
        num_groups, groups = NpClassifierObjective._assign_groups(
            target=TARGET, beta=BETA, scaled=SCALED, meta={"num_classes": COUNTS.shape[0]}
        )
        result = NpClassifierObjective._sample_candidates(
            num_groups=num_groups,
            groups=groups,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            random_state=RANDOM_STATE
        )
        self.assertEqual(result.shape, TARGET.shape)
        unique_groups, group_counts = np.unique(groups[result], return_counts=True)
        # group numbers and counts for groups with zero elements are not included
        ref_samples = NpClassifierObjective._get_group_samples(
            num_groups=num_groups,
            groups=groups,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION
        )
        active_groups = np.nonzero(ref_samples)[0]
        # find all groups with more than zero elements for comparison
        np.testing.assert_allclose(unique_groups, active_groups)
        np.testing.assert_allclose(group_counts, ref_samples[active_groups])

    def test_sample_candidates_2(self):
        result = NpClassifierObjective._sample_candidates(
            num_groups=NUM_LARGE_GROUPS,
            groups=LARGE_GROUPS,
            num_candidates=NUM_CANDIDATES_2,
            max_fraction=MAX_FRACTION_2,
            random_state=RANDOM_STATE
        )
        self.assertEqual(result.shape, LARGE_GROUPS.shape)
        unique_groups, group_counts = np.unique(LARGE_GROUPS[result], return_counts=True)
        np.testing.assert_allclose(unique_groups, np.arange(NUM_LARGE_GROUPS))
        # large_groups contains elements of all groups
        np.testing.assert_allclose(group_counts, SAMPLES_PER_LARGE_GROUP)

    def test_sample_candidates_3(self):
        num_groups_incomplete = 4
        groups_incomplete = np.array([0, 0, 0, 2, 2])
        result = NpClassifierObjective._sample_candidates(
            num_groups=num_groups_incomplete,
            groups=groups_incomplete,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            random_state=RANDOM_STATE
        )
        self.assertEqual(result.shape, groups_incomplete.shape)
        unique_groups, group_counts = np.unique(groups_incomplete[result], return_counts=True)
        # group numbers and counts for groups with zero elements are not included
        ref_samples = NpClassifierObjective._get_group_samples(
            num_groups=num_groups_incomplete,
            groups=groups_incomplete,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION
        )
        active_groups = np.nonzero(ref_samples)[0]
        # find all groups with more than zero elements for comparison
        np.testing.assert_allclose(unique_groups, active_groups)
        np.testing.assert_allclose(group_counts, ref_samples[active_groups])

    # method _log_group_breakdown() not tested as it deals with logging only

    def test_get_group_samples_1(self):
        num_groups, groups = NpClassifierObjective._assign_groups(
            target=TARGET, beta=BETA, scaled=SCALED, meta={"num_classes": COUNTS.shape[0]}
        )
        result = NpClassifierObjective._get_group_samples(
            num_groups=num_groups,
            groups=groups,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION
        )
        _, group_counts = np.unique(groups, return_counts=True)
        np.testing.assert_allclose(result, np.round(group_counts * MAX_FRACTION).astype(int))
        # all groups are too small so candidates are assigned based on max_fraction
        by_class = result[0::2] + result[1::2]  # adjacent group numbers indicate samples from the same class
        self.assertTrue(np.all(by_class > 0))
        self.assertTrue(np.all(by_class <= np.round(COUNTS * MAX_FRACTION)))

    @staticmethod
    def test_get_group_samples_2():
        result = NpClassifierObjective._get_group_samples(
            num_groups=NUM_LARGE_GROUPS,
            groups=LARGE_GROUPS,
            num_candidates=NUM_CANDIDATES_2,
            max_fraction=MAX_FRACTION_2
        )
        np.testing.assert_allclose(result, SAMPLES_PER_LARGE_GROUP)
        # group 0 and 2 split the remaining 410 samples after selecting the maximum of 30 % samples for groups 1 and 3

    @staticmethod
    def test_get_group_samples_3():
        result = NpClassifierObjective._get_group_samples(
            num_groups=4,
            groups=np.hstack([
                np.zeros(1000, dtype=int),
                np.ones(1000, dtype=int),
                2 * np.ones(1000, dtype=int),
                3 * np.ones(1000, dtype=int),
            ]),
            num_candidates=NUM_CANDIDATES_2,
            max_fraction=MAX_FRACTION_2
        )
        np.testing.assert_allclose(result, np.ones(4, dtype=int) * NUM_CANDIDATES_2 / 4)
        # all groups are large enough to support an equal split

    # methods _finalize_split() and _adjust_ref_weights() already tested by test_split_samples_1() above

    def test_get_starting_point_and_bounds_1(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        starting_point, bounds = objective.get_starting_point_and_bounds()
        shared.check_float_array(x=starting_point, name="starting_point")
        dimensions = objective._sample_data["cand_features"].shape
        self.assertEqual(starting_point.shape, (dimensions[1] + dimensions[0], ))
        np.testing.assert_allclose(
            starting_point[:dimensions[0]], START_FEATURE_WEIGHT * np.ones(dimensions[0]) / dimensions[0]
        )
        np.testing.assert_allclose(starting_point[dimensions[0]:], START_PROTOTYPE_WEIGHT * np.ones(dimensions[1]))
        self.assertEqual(len(bounds), dimensions[1] + dimensions[0])
        self.assertEqual(bounds[0][0], 0.0)
        self.assertEqual(bounds[0][1], np.inf)

    def test_evaluate_fail_1(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        dimension = objective._sample_data["cand_features"].shape[1] + objective._sample_data["cand_features"].shape[0]
        message = ""
        try:
            objective.evaluate(parameter=np.ones((dimension, 1)))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter parameter must be a 1D array.")

    def test_evaluate_fail_2(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        dimension = objective._sample_data["cand_features"].shape[1] + objective._sample_data["cand_features"].shape[0]
        message = ""
        try:
            objective.evaluate(parameter=np.ones(dimension - 1))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message,
            " ".join([
                "Parameter parameter must have as many elements as the number of features and candidates",
                "(expected {}, found {}).".format(dimension, dimension - 1)
            ])
        )

    def test_evaluate_1(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        dimensions = objective._sample_data["cand_features"].shape
        parameter = np.linspace(0.0, 1.0, dimensions[1] + dimensions[0])
        value, gradient = objective.evaluate(parameter)
        ref_value_1, ref_feature_gradient_1, ref_prototype_gradient_1 = objective._evaluate_penalty(
            feature_weights=parameter[:dimensions[1]],
            prototype_weights=parameter[dimensions[1]:],
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W
        )
        similarity = objective._compute_similarity(
            feature_weights=parameter[:dimensions[1]],
            sample_data=objective._sample_data
        )
        ref_value_2, ref_feature_gradient_2, ref_prototype_gradient_2 = objective._evaluate_likelihood(
            feature_weights=parameter[:dimensions[1]],
            prototype_weights=parameter[dimensions[1]:],
            sample_data=objective._sample_data,
            similarity=similarity,
            meta=objective._meta
        )
        self.assertAlmostEqual(value, ref_value_1 + ref_value_2)
        self.assertEqual(gradient.dtype, np.float64)  # solver expects 64bit
        np.testing.assert_allclose(gradient, np.hstack([
            ref_feature_gradient_1 + ref_feature_gradient_2, ref_prototype_gradient_1 + ref_prototype_gradient_2
        ]), atol=1e-6)

    def test_evaluate_2(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        dimensions = objective._sample_data["cand_features"].shape
        parameter = np.zeros(dimensions[1] + dimensions[0])
        # test that penalties are correctly applied even if optimization for sparseness is in effect
        value, gradient = objective.evaluate(parameter)
        ref_value_1, _, ref_prototype_gradient_1 = objective._evaluate_penalty(
            feature_weights=parameter[:dimensions[1]],
            prototype_weights=parameter[dimensions[1]:],
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W
        )
        similarity = objective._compute_similarity(
            feature_weights=parameter[:dimensions[1]],
            sample_data=objective._sample_data
        )
        ref_value_2, _, ref_prototype_gradient_2 = objective._evaluate_likelihood(
            feature_weights=parameter[:dimensions[1]],
            prototype_weights=parameter[dimensions[1]:],
            sample_data=objective._sample_data,
            similarity=similarity,
            meta=objective._meta
        )
        self.assertAlmostEqual(value, ref_value_1 + ref_value_2)
        self.assertEqual(gradient.dtype, np.float64)  # solver expects 64bit
        np.testing.assert_allclose(gradient, np.hstack([
            LAMBDA_V * (1.0 - ALPHA_V) * np.ones(dimensions[0]),
            ref_prototype_gradient_1 + ref_prototype_gradient_2
        ]), atol=1e-6)

    # methods _check_evaluate_parameter() and _evaluate_objective() already tested by the above

    def test_verify_sparseness_1(self):
        parameter = np.array([1.0, 0.0, 1.0, 1.0, 1.0], **shared.FLOAT_TYPE)
        # more than 70 % required for exploiting sparse structure
        result, active_values = NpClassifierObjective._verify_sparseness(parameter)
        shared.check_float_array(x=result, name="result")
        np.testing.assert_allclose(result, parameter)
        self.assertEqual(active_values, None)

    @staticmethod
    def test_verify_sparseness_2():
        parameter = np.array([1.0, 0.0, 1.0, 0.0, 1.0], **shared.FLOAT_TYPE)
        # less than 70 % required for exploiting sparse structure
        result, active_values = NpClassifierObjective._verify_sparseness(parameter)
        shared.check_float_array(x=result, name="result")
        np.testing.assert_allclose(result, np.ones(3))
        np.testing.assert_allclose(active_values, np.array([0, 2, 4]))

    def test_evaluate_penalty_1(self):
        feature_weights = np.array([0.0, 0.2, 0.4], **shared.FLOAT_TYPE)
        prototype_weights = np.array([0.6, 0.8], **shared.FLOAT_TYPE)
        value, feature_grad, prototype_grad = NpClassifierObjective._evaluate_penalty(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W
        )
        ref_value = LAMBDA_V * (
                0.5 * ALPHA_V * np.sum(feature_weights ** 2.0) + (1 - ALPHA_V) * np.sum(feature_weights)) + \
            LAMBDA_W * (
                0.5 * ALPHA_W * np.sum(prototype_weights ** 2.0) + (1 - ALPHA_W) * np.sum(prototype_weights))
        self.assertAlmostEqual(value, ref_value)
        shared.check_float_array(x=feature_grad, name="feature_grad")
        ref_feature_grad = LAMBDA_V * (ALPHA_V * feature_weights + 1 - ALPHA_V)
        np.testing.assert_allclose(feature_grad, ref_feature_grad)
        shared.check_float_array(x=prototype_grad, name="prototype_grad")
        ref_prototype_grad = LAMBDA_W * (ALPHA_W * prototype_weights + 1 - ALPHA_W)
        np.testing.assert_allclose(prototype_grad, ref_prototype_grad)

    def test_shrink_sample_data_1(self):
        sample_data = _get_consistent_example()["sample_data"]
        reduced_data, cache = NpClassifierObjective._shrink_sample_data(
            sample_data=sample_data,
            sample_cache=None,
            active_features=None
        )  # no reduced parameter vectors means no shrinkage
        self.assertEqual(sample_data, reduced_data)
        # this check works although sample_data contains numpy arrays, probably because the objects are identical
        self.assertEqual(cache, None)

    def test_shrink_sample_data_2(self):
        example = _get_consistent_example()
        active_features = _choose_half(example["feature_weights"])
        ref_cache = {
            "active_features": active_features,
            "active_prototypes": None,
            "sample_data": "test"  # use arbitrary payload as no operations are performed
        }
        reduced_data, cache = NpClassifierObjective._shrink_sample_data(
            sample_data=example["sample_data"],
            sample_cache=ref_cache,
            active_features=active_features
        )  # reduced feature weights and valid cache means cache is reused
        self.assertEqual(reduced_data, "test")
        self.assertEqual(cache, ref_cache)
        # this check works although sample_data contains numpy arrays, probably because the objects are identical

    def test_shrink_sample_data_3(self):
        example = _get_consistent_example()
        active_features = _choose_half(example["feature_weights"])
        reduced_data, cache = NpClassifierObjective._shrink_sample_data(
            sample_data=example["sample_data"],
            sample_cache=None,
            active_features=active_features
        )  # reduced feature weights and no valid cache means cache has to computed
        self.assertEqual(len(reduced_data), 13)
        shared.check_float_array(x=reduced_data["ref_features"], name="reduced_data['ref_features']")
        np.testing.assert_allclose(
            reduced_data["ref_features"], example["sample_data"]["ref_features"][:, active_features]
        )
        shared.check_float_array(x=reduced_data["ref_features_squared"], name="reduced_data['ref_features_squared']")
        np.testing.assert_allclose(
            reduced_data["ref_features_squared"], example["sample_data"]["ref_features_squared"][:, active_features]
        )
        np.testing.assert_allclose(reduced_data["ref_target"], example["sample_data"]["ref_target"])
        shared.check_float_array(x=reduced_data["ref_weights"], name="reduced_data['ref_weights']")
        np.testing.assert_allclose(reduced_data["ref_weights"], example["sample_data"]["ref_weights"])
        shared.check_float_array(x=reduced_data["ref_scale"], name="reduced_data['ref_scale']")
        np.testing.assert_allclose(reduced_data["ref_scale"], example["sample_data"]["ref_scale"])
        shared.check_float_array(x=reduced_data["ref_unscaled"], name="reduced_data['ref_unscaled']")
        np.testing.assert_allclose(reduced_data["ref_unscaled"], example["sample_data"]["ref_unscaled"])
        shared.check_float_array(x=reduced_data["cand_features"], name="reduced_data['cand_features']")
        np.testing.assert_allclose(
            reduced_data["cand_features"], example["sample_data"]["cand_features"][:, active_features]
        )
        shared.check_float_array(x=reduced_data["cand_features_squared"], name="reduced_data['cand_features_squared']")
        np.testing.assert_allclose(
            reduced_data["cand_features_squared"], example["sample_data"]["cand_features_squared"][:, active_features]
        )
        np.testing.assert_allclose(reduced_data["cand_target"], example["sample_data"]["cand_target"])
        np.testing.assert_allclose(reduced_data["cand_changes"], example["sample_data"]["cand_changes"])
        np.testing.assert_allclose(reduced_data["cand_index"], example["sample_data"]["cand_index"])
        self.assertEqual(reduced_data["shrink_sparse_features"], example["sample_data"]["shrink_sparse_features"])
        self.assertTrue(reduced_data["class_matches"].flags["F_CONTIGUOUS"])
        # boolean array must be F-contiguous so its product with another F-contiguous array retains the property
        np.testing.assert_allclose(reduced_data["class_matches"], example["sample_data"]["class_matches"])
        self.assertEqual(len(cache), 2)
        np.testing.assert_equal(cache["active_features"], active_features)
        self.assertEqual(cache["sample_data"], reduced_data)

    @staticmethod
    def test_compute_similarity_1():
        example = _get_consistent_example()
        # this function already calls NpClassifierObjective._compute_similarity()
        shared.check_float_array(x=example["similarity"], name="example[similarity']")
        scaled_reference = example["sample_data"]["ref_features"] * example["feature_weights"]
        scaled_prototypes = example["sample_data"]["cand_features"] * example["feature_weights"]
        reference_similarity = shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=scaled_prototypes,
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=np.sum(scaled_prototypes ** 2.0, axis=1)
        )
        np.testing.assert_allclose(example["similarity"], reference_similarity)

    def test_evaluate_likelihood_1(self):
        example = _get_consistent_example()
        value, feature_gradient, prototype_gradient = NpClassifierObjective._evaluate_likelihood(
            feature_weights=example["feature_weights"],
            prototype_weights=example["prototype_weights"],
            sample_data=example["sample_data"],
            similarity=example["similarity"],
            meta={"total_weight": example["total_weight"]}
        )
        shared_expressions = NpClassifierObjective._compute_shared_expressions(
            similarity=example["similarity"],
            sample_data=example["sample_data"],
            prototype_weights=example["prototype_weights"],
            meta={"total_weight": example["total_weight"]}
        )
        ref_value = NpClassifierObjective._compute_negative_log_likelihood(shared_expressions)
        ref_feature_gradient = NpClassifierObjective._compute_partial_feature_weights(
            shared_expressions=shared_expressions,
            feature_weights=example["feature_weights"]
        )
        ref_prototype_gradient = NpClassifierObjective._compute_partial_prototype_weights(shared_expressions)
        self.assertAlmostEqual(value, ref_value)
        shared.check_float_array(x=feature_gradient, name="feature_gradient")
        np.testing.assert_allclose(feature_gradient, ref_feature_gradient)
        shared.check_float_array(x=prototype_gradient, name="prototype_gradient")
        np.testing.assert_allclose(prototype_gradient, ref_prototype_gradient)

    @staticmethod
    def test_expand_gradient_1():
        feature_gradient = np.linspace(0.2, 1.0, 5).astype(**shared.FLOAT_TYPE)
        result = NpClassifierObjective._expand_gradient(
            gradient=feature_gradient,
            active_parameters=None,
            num_parameters=5
        )
        shared.check_float_array(x=result, name="result")
        np.testing.assert_allclose(result, feature_gradient)

    @staticmethod
    def test_expand_gradient_2():
        feature_gradient = np.linspace(0.2, 1.0, 5).astype(**shared.FLOAT_TYPE)
        active_features = np.array([1, 3, 5, 7, 9])
        result = NpClassifierObjective._expand_gradient(
            gradient=feature_gradient,
            active_parameters=active_features,
            num_parameters=10
        )
        shared.check_float_array(x=result, name="result")
        reference = np.zeros(10)
        reference[active_features] = feature_gradient
        np.testing.assert_allclose(result, reference)

    def test_get_batch_info_1(self):
        objective = NpClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            beta=BETA,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            random_state=RANDOM_STATE
        )
        dimensions = objective._sample_data["cand_features"].shape
        parameter = np.linspace(0.0, 1.0, dimensions[1] + dimensions[0])
        result = objective.get_batch_info(parameter)
        self.assertEqual(len(result), 5)
        shared.check_float_array(x=result["prototypes"], name="result['prototypes']")
        np.testing.assert_allclose(result["prototypes"], objective._sample_data["cand_features"])
        np.testing.assert_allclose(result["target"], objective._sample_data["cand_target"])
        shared.check_float_array(x=result["feature_weights"], name="result['feature_weights']")
        np.testing.assert_allclose(result["feature_weights"], parameter[:dimensions[1]])
        shared.check_float_array(x=result["prototype_weights"], name="result['prototype_weights']")
        np.testing.assert_allclose(result["prototype_weights"], parameter[dimensions[1]:])
        np.testing.assert_allclose(result["sample_index"], objective._sample_data["cand_index"])

    def test_compute_shared_expressions_1(self):
        example = _get_consistent_example()
        result = NpClassifierObjective._compute_shared_expressions(
            similarity=example["similarity"],
            sample_data=example["sample_data"],
            prototype_weights=example["prototype_weights"],
            meta={"total_weight": example["total_weight"]}
        )
        self.assertEqual(len(result), 12)
        shared.check_float_array(x=result["similarity"], name="result['similarity']")
        np.testing.assert_allclose(result["similarity"], example["similarity"])
        shared.check_float_array(x=result["similarity_matched"], name="result['similarity_matched']")
        np.testing.assert_allclose(
            result["similarity_matched"], example["similarity"] * example["sample_data"]["class_matches"]
        )
        shared.check_float_array(x=result["impact"], name="result['impact']")
        np.testing.assert_allclose(result["impact"], example["similarity"] * example["prototype_weights"])
        shared.check_float_array(x=result["impact_matched"], name="result['impact_matched']")
        np.testing.assert_allclose(
            result["impact_matched"],
            example["similarity"] * example["sample_data"]["class_matches"] * example["prototype_weights"]
        )
        shared.check_float_array(x=result["cand_features"], name="result['cand_features']")
        np.testing.assert_allclose(result["cand_features"], example["sample_data"]["cand_features"])
        shared.check_float_array(x=result["cand_features_squared"], name="result['cand_features_squared']")
        np.testing.assert_allclose(result["cand_features_squared"], example["sample_data"]["cand_features_squared"])
        shared.check_float_array(x=result["ref_features"], name="result['ref_features']")
        np.testing.assert_allclose(result["ref_features"], example["sample_data"]["ref_features"])
        shared.check_float_array(x=result["ref_features_squared"], name="result['ref_features_squared']")
        np.testing.assert_allclose(result["ref_features_squared"], example["sample_data"]["ref_features_squared"])
        shared.check_float_array(x=result["ref_unscaled"], name="result['ref_unscaled']")
        np.testing.assert_allclose(result["ref_unscaled"], example["ref_unscaled"])
        shared.check_float_array(x=result["ref_scale"], name="result['ref_scale']")
        np.testing.assert_allclose(result["ref_scale"], example["ref_scale"])
        shared.check_float_array(x=result["ref_weights"], name="result['ref_weights']")
        np.testing.assert_allclose(result["ref_weights"], example["sample_data"]["ref_weights"])
        self.assertEqual(result["total_weight"], example["total_weight"])

    def test_compute_negative_log_likelihood_1(self):
        example = _get_consistent_example()
        shared_expressions = NpClassifierObjective._compute_shared_expressions(
            similarity=example["similarity"],
            sample_data=example["sample_data"],
            prototype_weights=example["prototype_weights"],
            meta={"total_weight": example["total_weight"]}
        )
        result = NpClassifierObjective._compute_negative_log_likelihood(shared_expressions)
        reference = -1.0 * np.sum(
            np.log(shared_expressions["ref_unscaled"] / shared_expressions["ref_scale"]) *
            shared_expressions["ref_weights"]
        ) / shared_expressions["total_weight"]
        self.assertAlmostEqual(result, reference, delta=1e-6)

    @staticmethod
    def test_compute_partial_feature_weights_1():
        example = _get_consistent_example()
        shared_expressions = NpClassifierObjective._compute_shared_expressions(
            similarity=example["similarity"],
            sample_data=example["sample_data"],
            prototype_weights=example["prototype_weights"],
            meta={"total_weight": example["total_weight"]}
        )
        result = NpClassifierObjective._compute_partial_feature_weights(
            shared_expressions=shared_expressions,
            feature_weights=example["feature_weights"]
        )
        shared.check_float_array(x=result, name="result")
        reference = np.zeros_like(example["feature_weights"])
        for i in range(example["sample_data"]["cand_features"].shape[0]):
            delta = (example["sample_data"]["ref_features"] - example["sample_data"]["cand_features"][i]) ** 2.0
            reference += np.sum(
                delta.transpose() * shared_expressions["impact_matched"][:, i] * shared_expressions["ref_weights"] /
                shared_expressions["ref_unscaled"], axis=1
            )
            reference -= np.sum(
                delta.transpose() * shared_expressions["impact"][:, i] * shared_expressions["ref_weights"] /
                shared_expressions["ref_scale"], axis=1
            )
        reference *= example["feature_weights"] / shared_expressions["total_weight"]
        np.testing.assert_allclose(result, reference, atol=1e-6)

    @staticmethod
    def test_compute_partial_feature_weights_2():
        empty_vector = np.zeros(0, **shared.FLOAT_TYPE)
        result = NpClassifierObjective._compute_partial_feature_weights(
            shared_expressions=None,  # value does not matter in this case
            feature_weights=np.zeros(0, **shared.FLOAT_TYPE)
        )
        shared.check_float_array(x=result, name="result")
        np.testing.assert_equal(result, empty_vector)

    @staticmethod
    def test_quick_compute_part_1():
        example = _get_consistent_example()
        shared_expressions = NpClassifierObjective._compute_shared_expressions(
            similarity=example["similarity"],
            sample_data=example["sample_data"],
            prototype_weights=example["prototype_weights"],
            meta={"total_weight": example["total_weight"]}
        )
        result = NpClassifierObjective._quick_compute_part(
            ref_features=shared_expressions["ref_features"],
            ref_features_squared=shared_expressions["ref_features_squared"],
            cand_features=shared_expressions["cand_features"],
            cand_features_squared=shared_expressions["cand_features_squared"],
            impact=shared_expressions["impact"]
        )
        shared.check_float_array(x=result, name="result")
        reference = np.zeros_like(shared_expressions["ref_features"])
        for i in range(shared_expressions["ref_features"].shape[0]):
            for d in range(shared_expressions["ref_features"].shape[1]):
                for j in range(shared_expressions["cand_features"].shape[0]):
                    reference[i, d] += (
                        shared_expressions["ref_features"][i, d] - shared_expressions["cand_features"][j, d]
                    ) ** 2.0 * shared_expressions["impact"][i, j]
        np.testing.assert_allclose(result, reference, atol=1e-5)
        # the naive implementation above tends to give a result that differs from the efficient implementation on the
        # order of 1e-6 in absolute value

    @staticmethod
    def test_compute_partial_prototype_weights_1():
        example = _get_consistent_example()
        shared_expressions = NpClassifierObjective._compute_shared_expressions(
            similarity=example["similarity"],
            sample_data=example["sample_data"],
            prototype_weights=example["prototype_weights"],
            meta={"total_weight": example["total_weight"]}
        )
        result = NpClassifierObjective._compute_partial_prototype_weights(shared_expressions)
        shared.check_float_array(x=result, name="result")
        reference = (np.sum(
            shared_expressions["similarity"].transpose() * shared_expressions["ref_weights"] /
            shared_expressions["ref_scale"], axis=1
        ) - np.sum(
            shared_expressions["similarity_matched"].transpose() * shared_expressions["ref_weights"] /
            shared_expressions["ref_unscaled"], axis=1
        )) / shared_expressions["total_weight"]
        np.testing.assert_allclose(result, reference, atol=1e-6)
