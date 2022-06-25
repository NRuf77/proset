"""Unit tests for code in proset.objectives that relies on tensorflow for evaluation.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from copy import deepcopy
from unittest import TestCase

import numpy as np
import tensorflow as tf

from proset.objectives.tf_classifier_objective import TfClassifierObjective
import proset.shared as shared

# pylint: disable=wrong-import-order
from test.test_shared_classifier import MockSetManager, _get_consistent_example, FEATURES, TARGET, COUNTS, WEIGHTS, \
    UNSCALED, SCALE, NUM_CANDIDATES, MAX_FRACTION, LAMBDA_V, LAMBDA_W, ALPHA_V, ALPHA_W, RANDOM_STATE


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestTfClassifierObjective(TestCase):
    """Unit tests for class NpClassifierObjective.

    The tests also cover abstract superclasses Objective and TfObjective.
    """

    def test_init_fail_1(self):
        # test only one exception to ensure relevant functions are called; remaining exceptions are tested already
        # in test_np_objective.py
        message = ""
        try:
            TfClassifierObjective(
                features=FEATURES,
                target=TARGET.astype(float),
                weights=WEIGHTS,
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
        self.assertEqual(message, "Parameter target must be an integer array.")

    def test_init_1(self):
        objective = TfClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
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
        np.testing.assert_allclose(objective._meta["counts"], COUNTS)
        dimensions = objective._sample_data["cand_features"].shape
        self.assertEqual(objective._meta["num_features"], dimensions[1])
        self.assertEqual(objective._meta["num_parameters"], dimensions[1] + dimensions[0])
        self.assertEqual(objective._meta["total_weight"], np.sum(WEIGHTS))
        self.assertEqual(len(objective._sample_data), 11)  # a full check is performed by test_finalize_split() below
        self.assertEqual(objective._lambda_v, LAMBDA_V)
        self.assertEqual(objective._lambda_w, LAMBDA_W)
        self.assertEqual(objective._alpha_v, ALPHA_V)
        self.assertEqual(objective._alpha_w, ALPHA_W)

    def test_init_2(self):
        objective = TfClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=None,  # use unit weights
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

    # method _check_init_values() already tested by the above or via test_np_objective.py

    def test_split_samples_1(self):
        result = TfClassifierObjective._split_samples(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
            num_candidates=NUM_CANDIDATES,
            max_fraction=MAX_FRACTION,
            set_manager=MockSetManager(),
            random_state=RANDOM_STATE,
            meta={"counts": COUNTS}
        )
        self.assertEqual(len(result), 11)
        candidates = np.zeros(FEATURES.shape[0], dtype=bool)
        candidates[result["cand_index"]] = True  # need to reconstruct candidates from result due to randomization
        reference = np.logical_not(candidates)
        np.testing.assert_allclose(result["ref_features"].numpy(), FEATURES[reference])
        np.testing.assert_allclose(result["ref_features_squared"].numpy(), FEATURES[reference] ** 2.0)
        ref_target = TARGET[reference]
        np.testing.assert_allclose(result["ref_target"].numpy(), ref_target)
        class_scales = np.zeros(max(TARGET) + 1, dtype=float)
        for i in range(class_scales.shape[0]):
            ix = TARGET == i
            class_scales[i] = np.sum(WEIGHTS[ix]) / np.sum(WEIGHTS[np.logical_and(ix, reference)])
        ref_weights = WEIGHTS[reference]
        for i, target in enumerate(ref_target):
            ref_weights[i] *= class_scales[target]
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.sum(ref_weights), np.sum(WEIGHTS), delta=1e-6)
        # check whether reference weights are consistent
        np.testing.assert_allclose(result["ref_weights"].numpy(), ref_weights, atol=1e-6)
        ref_unscaled = UNSCALED[reference]
        ref_unscaled = np.array([ref_unscaled[i, ref_target[i]] for i in range(ref_unscaled.shape[0])])
        np.testing.assert_allclose(result["ref_unscaled"].numpy(), ref_unscaled)
        np.testing.assert_allclose(result["ref_scale"].numpy(), SCALE[reference])
        np.testing.assert_allclose(result["cand_features"].numpy(), FEATURES[candidates])
        np.testing.assert_allclose(result["cand_features_squared"].numpy(), FEATURES[candidates] ** 2.0)
        np.testing.assert_allclose(result["cand_target"].numpy(), TARGET[candidates])
        np.testing.assert_allclose(result["cand_index"], np.nonzero(candidates)[0])
        class_matches = np.zeros((result["ref_features"].shape[0], result["cand_features"].shape[0]), dtype=float)
        for i in range(result["ref_features"].shape[0]):
            for j in range(result["cand_features"].shape[0]):
                class_matches[i, j] = result["ref_target"][i] == result["cand_target"][j]
        np.testing.assert_allclose(result["class_matches"].numpy(), class_matches)

    def test_assign_groups_1(self):
        num_groups, groups = TfClassifierObjective._assign_groups(
            target=TARGET,
            unscaled=UNSCALED,
            scale=SCALE,
            meta={"counts": COUNTS}
        )
        self.assertEqual(num_groups, 6)
        reference = 2 * TARGET
        prediction = np.argmax(UNSCALED, axis=1)
        reference[prediction != TARGET] += 1
        np.testing.assert_allclose(groups, reference)

    # method _sample_candidates() already tested by the above or via test_np_objective.py

    # method _log_group_breakdown() not tested as it deals with logging only

    # method _get_group_samples() already tested by the above or via test_np_objective.py

    # method _finalize_split() already tested by test_split_samples_1() above

    def test_convert_sample_data_1(self):
        vector = np.array([1, 2, 3])
        sample_data = {"vector": vector, "exclude": "exclude"}
        TfClassifierObjective._convert_sample_data(sample_data=sample_data, exclude=["exclude"])
        # sample_data is updated in place
        self.assertEqual(len(sample_data), 2)
        # noinspection PyUnresolvedReferences
        np.testing.assert_array_equal(sample_data["vector"].numpy(), vector)  # pylint: disable=no-member
        self.assertEqual(sample_data["exclude"], "exclude")

    # method get_starting_point_and_bounds() already tested via test_np_objective.py

    def test_evaluate_fail_1(self):
        # test only one exception to ensure relevant functions are called; remaining exceptions are tested already
        # in test_np_objective.py
        objective = TfClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
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

    def test_evaluate_1(self):
        objective = TfClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
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
        parameter = np.linspace(0.0, 1.0, dimensions[1] + dimensions[0]).astype(**shared.FLOAT_TYPE)
        value, gradient = objective.evaluate(parameter)
        feature_weights = tf.Variable(parameter[:dimensions[1]], dtype=float)
        prototype_weights = tf.Variable(parameter[dimensions[1]:], dtype=float)
        ref_value, ref_gradient = objective._make_gradient_tape(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            sample_data=objective._sample_data,
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            meta=objective._meta
        )
        ref_gradient = ref_gradient.gradient(target=ref_value, sources=[feature_weights, prototype_weights])
        self.assertTrue(isinstance(value, np.float64))
        np.testing.assert_allclose(value, ref_value.numpy())
        self.assertEqual(gradient.dtype, np.float64)  # solver expects 64bit
        np.testing.assert_allclose(gradient, np.hstack([ref_gradient[0].numpy(), ref_gradient[1].numpy()]))

    # methods _check_evaluate_parameter() and _evaluate_objective() already tested by the above

    def test_make_gradient_tape_1(self):
        example = _get_consistent_example()
        sample_data = deepcopy(example["sample_data"])
        sample_data.pop("cand_changes")
        sample_data.pop("shrink_sparse_features")
        ref_unscaled = np.array([
            sample_data["ref_unscaled"][i, sample_data["ref_target"][i]]
            for i in range(sample_data["ref_unscaled"].shape[0])
        ])
        sample_data["ref_unscaled"] = ref_unscaled.copy()
        TfClassifierObjective._convert_sample_data(sample_data=sample_data, exclude=["cand_index"])
        feature_weights = tf.Variable(example["feature_weights"], dtype=float)
        prototype_weights = tf.Variable(example["prototype_weights"], dtype=float)
        objective, gradient_tape = TfClassifierObjective._make_gradient_tape(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            sample_data=sample_data,
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W,
            meta={"total_weight": example["total_weight"]}
        )
        gradient = gradient_tape.gradient(target=objective, sources=[feature_weights, prototype_weights])
        impact = example["similarity"] * example["prototype_weights"]
        impact_matched = impact * example["sample_data"]["class_matches"]
        ref_unscaled += np.sum(impact_matched, axis=1)
        scale = np.sum(impact, axis=1) + example["sample_data"]["ref_scale"]
        ref_objective = -np.sum(
            np.log(ref_unscaled / scale + shared.LOG_OFFSET) * example["sample_data"]["ref_weights"]
        ) / example["total_weight"]
        ref_objective += TfClassifierObjective._evaluate_penalty(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W
        ).numpy()
        self.assertAlmostEqual(objective.numpy(), ref_objective, places=6)
        reference = np.zeros_like(example["feature_weights"])
        for i in range(example["sample_data"]["cand_features"].shape[0]):
            delta = (example["sample_data"]["ref_features"] - example["sample_data"]["cand_features"][i]) ** 2.0
            reference += np.sum(
                delta.transpose() * impact_matched[:, i] * example["sample_data"]["ref_weights"] / ref_unscaled, axis=1
            )
            reference -= np.sum(
                delta.transpose() * impact[:, i] * example["sample_data"]["ref_weights"] / scale, axis=1
            )
        reference *= example["feature_weights"] / example["total_weight"]
        reference += LAMBDA_V * (ALPHA_V * example["feature_weights"] + (1 - ALPHA_V))
        np.testing.assert_allclose(gradient[0].numpy(), reference, atol=1e-6)
        reference = (
            np.sum(example["similarity"].transpose() * example["sample_data"]["ref_weights"] / scale, axis=1) -
            np.sum(
                (example["similarity"] * example["sample_data"]["class_matches"]).transpose() *
                example["sample_data"]["ref_weights"] / ref_unscaled, axis=1
            )
        ) / example["total_weight"]
        reference += LAMBDA_W * (ALPHA_W * example["prototype_weights"] + (1 - ALPHA_W))
        np.testing.assert_allclose(gradient[1].numpy(), reference, atol=1e-6)

    @staticmethod
    def test_quick_compute_similarity_1():
        example = _get_consistent_example()
        reference = example["sample_data"]["ref_features"]
        prototypes = example["sample_data"]["cand_features"]
        feature_weights = example["feature_weights"]
        scaled_reference = tf.constant(reference, dtype=float) * tf.Variable(feature_weights, dtype=float)
        scaled_prototypes = tf.constant(prototypes, dtype=float) * tf.Variable(feature_weights, dtype=float)
        similarity = TfClassifierObjective._quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=scaled_prototypes,
            ssq_reference=tf.reduce_sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=tf.reduce_sum(scaled_prototypes ** 2.0, axis=1)
        )
        reference_similarity = np.zeros((reference.shape[0], prototypes.shape[0]), dtype=float)
        for i in range(reference.shape[0]):
            for j in range(prototypes.shape[0]):
                reference_similarity[i, j] = np.exp(
                    -0.5 * np.sum(((reference[i] - prototypes[j]) * feature_weights) ** 2.0)
                )
        np.testing.assert_allclose(similarity.numpy(), reference_similarity, atol=1e-6)

    def test_evaluate_penalty_1(self):
        feature_weights = np.array([0.0, 0.2, 0.4])
        prototype_weights = np.array([0.6, 0.8])
        value = TfClassifierObjective._evaluate_penalty(
            feature_weights=tf.Variable(feature_weights, dtype=float),
            prototype_weights=tf.Variable(prototype_weights, dtype=float),
            lambda_v=LAMBDA_V,
            lambda_w=LAMBDA_W,
            alpha_v=ALPHA_V,
            alpha_w=ALPHA_W
        )
        ref_value = LAMBDA_V * (
            ALPHA_V * np.sum(feature_weights ** 2.0) / 2.0 + (1 - ALPHA_V) * np.sum(feature_weights)
        ) + LAMBDA_W * (
            ALPHA_W * np.sum(prototype_weights ** 2.0) / 2.0 + (1 - ALPHA_W) * np.sum(prototype_weights)
        )
        self.assertAlmostEqual(value.numpy(), ref_value)  # scalar tensor evaluates to float

    def test_get_batch_info_1(self):
        objective = TfClassifierObjective(
            features=FEATURES,
            target=TARGET,
            weights=WEIGHTS,
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
        self.assertTrue(np.issubdtype(result["target"].dtype, np.integer))
        np.testing.assert_allclose(result["target"], objective._sample_data["cand_target"])
        shared.check_float_array(x=result["feature_weights"], name="result['feature_weights']")
        np.testing.assert_allclose(result["feature_weights"], parameter[:dimensions[1]])
        shared.check_float_array(x=result["prototype_weights"], name="result['prototype_weights']")
        np.testing.assert_allclose(result["prototype_weights"], parameter[dimensions[1]:])
        np.testing.assert_allclose(result["sample_index"], objective._sample_data["cand_index"])
