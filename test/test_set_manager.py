"""Unit tests for code in set_manager.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

from proset.set_manager import ClassifierSetManager


# define common objects for testing
TARGET = np.array([0, 1, 0, 2, 1, 2])
_, MARGINALS = np.unique(TARGET, return_counts=True)
MARGINALS = MARGINALS / np.sum(MARGINALS)
PROTOTYPES = np.array([
    [1.0, 0.0, 0.0, 2.7],
    [1.0, 0.0, 0.0, 1.8],
    [0.0, 1.0, 0.0, -3.5],
    [0.0, 1.0, 0.0, -1.9],
    [0.0, 0.0, 1.0, 12.0],
    [0.0, 0.0, 1.0, 8.0]
])
FEATURE_WEIGHTS = np.array([0.5, 0.0, 1.5, 0.1])
PROTOTYPE_WEIGHTS = np.array([1.0, 2.0, 0.0, 2.0, 1.0, 0.5])
SAMPLE_INDEX = np.array([4, 7, 11, 15, 27, 40])
REFERENCE = np.array([
    [1.0, 0.0, 0.0, 3.0],
    [0.0, 1.0, 0.0, -2.0],
    [0.0, 0.0, 1.0, 9.5]
])
BATCH_INFO = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": FEATURE_WEIGHTS,
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_NO_FEATURES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": np.zeros_like(FEATURE_WEIGHTS),
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_NO_PROTOTYPES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": FEATURE_WEIGHTS,
    "prototype_weights": np.zeros_like(PROTOTYPE_WEIGHTS),
    "sample_index": SAMPLE_INDEX
}


class TestClassifierSetManager(TestCase):
    """Unit tests class ClassifierSetManager.

    The tests also cover abstract superclass SetManager.
    """

    def test_init_fail_1(self):
        message = ""
        try:
            ClassifierSetManager(target=np.array([0, 2]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "The classes must be encoded as integers from 0 to K - 1 and each class must be present."
        )

    def test_init_1(self):
        manager = ClassifierSetManager(target=TARGET)
        self.assertEqual(manager.num_batches, 0)
        self.assertEqual(manager.num_features, None)
        np.testing.assert_allclose(manager.marginals, MARGINALS)

    # method _get_baseline_distribution() already tested by the above

    def test_add_batch_fail_1(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": np.array([1.0]),
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must have integer elements.")

    def test_add_batch_fail_2(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": np.array([-1]),
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter target must encode the classes as integers from 0 to K - 1."
        )

    def test_add_batch_fail_3(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": np.array([3]),
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter target must encode the classes as integers from 0 to K - 1."
        )

    def test_add_batch_fail_4(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES[:, 0],
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter prototypes must be a 2D array.")

    def test_add_batch_fail_5(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch({
            "prototypes": PROTOTYPES,
            "target": TARGET,
            "feature_weights": FEATURE_WEIGHTS,
            "prototype_weights": np.zeros_like(PROTOTYPE_WEIGHTS),
            "sample_index": SAMPLE_INDEX
        })  # a batch with all prototype weights equal to 0.0 still counts, although the content is None
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES[:, :-1],
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS[:-1],
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter prototypes has {} columns but {} are expected.".format(
            PROTOTYPES.shape[1] - 1, PROTOTYPES.shape[1]
        ))

    def test_add_batch_fail_6(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET[:, np.newaxis],
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must be a 1D array.")

    def test_add_batch_fail_7(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET[:(PROTOTYPES.shape[0] - 1)],
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must have as many elements as prototypes has rows.")

    def test_add_batch_fail_8(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS[:, np.newaxis],
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter feature_weights must be a 1D array.")

    def test_add_batch_fail_9(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS[:(PROTOTYPES.shape[1] - 1)],
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter feature_weights must have as many elements as prototypes has columns.")

    def test_add_batch_fail_10(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS[:, np.newaxis],
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter prototype_weights must be a 1D array.")

    def test_add_batch_fail_11(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS[:(PROTOTYPES.shape[0] - 1)],
                "sample_index": SAMPLE_INDEX
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter prototype_weights must have as many elements as prototypes has rows.")

    def test_add_batch_fail_12(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX[:, np.newaxis]
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter sample_index must be a 1D array.")

    def test_add_batch_fail_13(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX[:(PROTOTYPES.shape[0] - 1)]
            })
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter sample_index must have as many elements as prototypes has rows.")

    def test_add_batch_1(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        self.assertEqual(manager.num_batches, 1)
        self.assertEqual(manager.num_features, len(FEATURE_WEIGHTS))

    def test_add_batch_3(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        self.assertEqual(manager.num_batches, 2)

    # method _check_batch() already tested by the above

    def test_process_batch_1(self):
        result = ClassifierSetManager._process_batch(BATCH_INFO_NO_PROTOTYPES)
        # no active prototypes means None is returned since there is no effect on the model
        self.assertEqual(result, None)

    def test_process_batch_2(self):
        result = ClassifierSetManager._process_batch(BATCH_INFO_NO_FEATURES)
        # a batch can be added with no active features to define a global correction
        self.assertEqual(len(result), 7)
        np.testing.assert_allclose(result["active_features"], np.zeros(0))
        unique_target = np.unique(TARGET)
        # global correction is consolidated to have only one prototype per distinct target value
        unique_weights = np.zeros_like(unique_target, dtype=float)
        unique_index = np.zeros_like(unique_target)
        for i, value in enumerate(unique_target):
            unique_weights[i] = np.sum(PROTOTYPE_WEIGHTS[TARGET == value])
            unique_index[i] = np.min(SAMPLE_INDEX[TARGET == value])
        np.testing.assert_allclose(result["scaled_prototypes"], np.zeros((unique_target.shape[0], 0)))
        np.testing.assert_allclose(result["ssq_prototypes"], np.zeros(unique_target.shape[0]))
        np.testing.assert_allclose(result["target"], unique_target)
        np.testing.assert_allclose(result["feature_weights"], np.zeros(0))
        np.testing.assert_allclose(result["prototype_weights"], unique_weights)
        np.testing.assert_allclose(result["sample_index"], unique_index)

    def test_process_batch_3(self):
        result = ClassifierSetManager._process_batch(BATCH_INFO)
        self.assertEqual(len(result), 7)
        active_features = np.where(FEATURE_WEIGHTS > 0.0)[0]
        np.testing.assert_allclose(result["active_features"], active_features)
        active_prototypes = np.where(PROTOTYPE_WEIGHTS > 0.0)[0]
        scaled_prototypes = PROTOTYPES[:, active_features][active_prototypes, :] * FEATURE_WEIGHTS[active_features]
        np.testing.assert_allclose(result["scaled_prototypes"], scaled_prototypes)
        np.testing.assert_allclose(result["ssq_prototypes"], np.sum(scaled_prototypes ** 2.0, axis=1))
        np.testing.assert_allclose(result["target"], TARGET[active_prototypes])
        np.testing.assert_allclose(result["feature_weights"], FEATURE_WEIGHTS[active_features])
        np.testing.assert_allclose(result["prototype_weights"], PROTOTYPE_WEIGHTS[active_prototypes])
        np.testing.assert_allclose(result["sample_index"], SAMPLE_INDEX[active_prototypes])

    def test_evaluate_unscaled_fail_1(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE[:, 0], num_batches=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features must be a 2D array.")

    def test_evaluate_unscaled_fail_2(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        message = ""
        try:
            manager.evaluate_unscaled(features=np.zeros((3, 3)), num_batches=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features has 3 columns but {} are expected.".format(PROTOTYPES.shape[1]))

    def test_evaluate_unscaled_fail_3(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE, num_batches=np.array([-1]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must not contain negative values if passing a vector.")

    def test_evaluate_unscaled_fail_4(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE, num_batches=np.array([1]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message,
            "Parameter num_batches must not contain values greater than the available number of 0 if passing a vector."
        )

    def test_evaluate_unscaled_fail_5(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE, num_batches=np.array([1, 0]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter num_batches must contain strictly increasing elements if passing a vector."
        )

    def test_evaluate_unscaled_fail_6(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE, num_batches=-1)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must not be negative.")

    def test_evaluate_unscaled_fail_7(self):
        manager = ClassifierSetManager(target=TARGET)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE, num_batches=1)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must be less than or equal to the available number of 0.")

    def test_evaluate_unscaled_1(self):
        manager = ClassifierSetManager(target=TARGET)
        unscaled = manager.evaluate_unscaled(features=REFERENCE, num_batches=None)
        # no batches means marginal distribution is returned
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_evaluate_unscaled_2(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO_NO_PROTOTYPES)  # batch with no prototypes has no impact on the model
        unscaled = manager.evaluate_unscaled(features=REFERENCE, num_batches=0)
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_evaluate_unscaled_3(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        unscaled = manager.evaluate_unscaled(REFERENCE, num_batches=np.array([0]))  # evaluate marginals only
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_evaluate_unscaled_4(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        unscaled = manager.evaluate_unscaled(REFERENCE, num_batches=None)
        self.assertEqual(len(unscaled), 1)
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, _ = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        for i in range(3):
            keep = target == i
            ref_unscaled[:, i] += np.sum(impact[:, keep], axis=1)
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.sum(ref_unscaled, axis=1))

    def test_evaluate_unscaled_5(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        unscaled = manager.evaluate_unscaled(REFERENCE, num_batches=np.array([0, 2]))
        self.assertEqual(len(unscaled), 2)
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, _ = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch, batch],
            num_batches=2,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        ref_batch_0 = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        ref_batch_2 = ref_batch_0.copy()
        for i in range(3):
            keep = target == i
            ref_batch_2[:, i] += np.sum(impact[:, keep], axis=1)
        np.testing.assert_allclose(unscaled[0][0], ref_batch_0)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_batch_0.shape[0]))
        np.testing.assert_allclose(unscaled[1][0], ref_batch_2)
        np.testing.assert_allclose(unscaled[1][1], np.sum(ref_batch_2, axis=1))

    # method _check_evaluate_input() already tested by the above

    def test_compute_impact_1(self):
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[],
            num_batches=0,
            meta={}
        )  # check default values for set manager with no data
        np.testing.assert_allclose(impact, np.zeros((REFERENCE.shape[0], 0)))
        np.testing.assert_allclose(target, np.zeros(0))
        np.testing.assert_allclose(batch_index, np.zeros(0))

    def test_compute_impact_2(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO_NO_FEATURES)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1]}
        )  # batch with no feature weights define a global correction
        unique_targets = np.unique(TARGET)
        # global correction is consolidated to have only one prototype per distinct target value
        combined_weights = np.zeros_like(unique_targets, dtype=float)
        for i, value in enumerate(unique_targets):
            combined_weights[i] = np.sum(PROTOTYPE_WEIGHTS[TARGET == value])
        np.testing.assert_allclose(impact, np.tile(combined_weights, (REFERENCE.shape[0], 1)))
        np.testing.assert_allclose(target, unique_targets)
        np.testing.assert_allclose(batch_index, np.zeros(unique_targets.shape[0], dtype=int))

    def test_compute_impact_3(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO_NO_PROTOTYPES)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={}
        )  # batch with no prototypes has no impact on the model
        np.testing.assert_allclose(impact, np.zeros((REFERENCE.shape[0], 0)))
        np.testing.assert_allclose(target, np.zeros(0))
        np.testing.assert_allclose(batch_index, np.zeros(0))

    def test_compute_impact_4(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=0,
            meta={}
        )  # check default values for set manager with data if num_batches is set to 0
        np.testing.assert_allclose(impact, np.zeros((REFERENCE.shape[0], 0)))
        np.testing.assert_allclose(target, np.zeros(0))
        np.testing.assert_allclose(batch_index, np.zeros(0))

    def test_compute_impact_5(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        active_prototypes = np.where(PROTOTYPE_WEIGHTS > 0.0)[0]
        active_features = np.where(FEATURE_WEIGHTS > 0.0)[0]
        ref_impact = np.zeros((REFERENCE.shape[0], len(active_prototypes)))
        for i in range(REFERENCE.shape[0]):
            for j in range(len(active_prototypes)):
                ref_impact[i, j] = np.exp(
                    -0.5 * np.sum(((REFERENCE[i, active_features] - PROTOTYPES[active_prototypes[j], active_features])
                                   * FEATURE_WEIGHTS[active_features]) ** 2.0)
                )
        ref_impact = ref_impact * PROTOTYPE_WEIGHTS[active_prototypes]
        np.testing.assert_allclose(impact, ref_impact)
        np.testing.assert_allclose(target, TARGET[active_prototypes])
        np.testing.assert_allclose(batch_index, np.zeros(active_prototypes.shape[0], dtype=int))

    def test_compute_impact_6(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch, batch],
            num_batches=2,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        active_prototypes = np.where(PROTOTYPE_WEIGHTS > 0.0)[0]
        active_features = np.where(FEATURE_WEIGHTS > 0.0)[0]
        ref_impact = np.zeros((REFERENCE.shape[0], len(active_prototypes)))
        for i in range(REFERENCE.shape[0]):
            for j in range(len(active_prototypes)):
                ref_impact[i, j] = np.exp(
                    -0.5 * np.sum(
                        ((REFERENCE[i, active_features] - PROTOTYPES[active_prototypes[j], active_features])
                         * FEATURE_WEIGHTS[active_features]) ** 2.0)
                )
        ref_impact = ref_impact * PROTOTYPE_WEIGHTS[active_prototypes]
        np.testing.assert_allclose(impact, np.hstack([ref_impact, ref_impact]))
        np.testing.assert_allclose(target, np.hstack([TARGET[active_prototypes], TARGET[active_prototypes]]))
        np.testing.assert_allclose(batch_index, np.hstack([
            np.zeros(active_prototypes.shape[0], dtype=int), np.ones(active_prototypes.shape[0], dtype=int)
        ]))

    def test_convert_to_unscaled_1(self):
        unscaled = ClassifierSetManager._convert_to_unscaled(
            impact=np.zeros((REFERENCE.shape[0], 0)),
            target=np.zeros(0, dtype=int),
            batch_index=np.zeros(0, dtype=int),
            num_batches=0,
            meta={"marginals": MARGINALS}
        )  # check handling of defaults for no data or no batches being evaluated
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_convert_to_unscaled_2(self):
        unscaled = ClassifierSetManager._convert_to_unscaled(
            impact=np.zeros((REFERENCE.shape[0], 0)),
            target=np.zeros(0, dtype=int),
            batch_index=np.zeros(0, dtype=int),
            num_batches=np.array([0]),
            meta={"marginals": MARGINALS}
        )  # as above, passing number of batches as array
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_convert_to_unscaled_3(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        unscaled = ClassifierSetManager._convert_to_unscaled(
            impact=impact,
            target=target,
            batch_index=batch_index,
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1], "marginals": MARGINALS}
        )
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        for i in range(3):
            keep = target == i
            ref_unscaled[:, i] += np.sum(impact[:, keep], axis=1)
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.sum(ref_unscaled, axis=1))

    def test_convert_to_unscaled_4(self):
        batch = ClassifierSetManager._process_batch({
            "prototypes": PROTOTYPES,
            "target": np.array([0, 2, 0, 2, 0, 2]),
            "feature_weights": FEATURE_WEIGHTS,
            "prototype_weights": PROTOTYPE_WEIGHTS,
            "sample_index": SAMPLE_INDEX
        })  # test the special case where not all classes are present in a batch
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        unscaled = ClassifierSetManager._convert_to_unscaled(
            impact=impact,
            target=target,
            batch_index=batch_index,
            num_batches=np.array([1]),  # test passing a single value as array
            meta={"num_features": PROTOTYPES.shape[1], "marginals": MARGINALS}
        )
        self.assertEqual(len(unscaled), 1)
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        for i in range(3):
            keep = target == i
            ref_unscaled[:, i] += np.sum(impact[:, keep], axis=1)
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        np.testing.assert_allclose(unscaled[0][1], np.sum(ref_unscaled, axis=1))

    def test_convert_to_unscaled_5(self):
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, batch_index = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch, batch],
            num_batches=2,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        unscaled = ClassifierSetManager._convert_to_unscaled(
            impact=impact,
            target=target,
            batch_index=batch_index,
            num_batches=np.array([0, 2]),
            meta={"num_features": PROTOTYPES.shape[1], "marginals": MARGINALS}
        )
        self.assertEqual(len(unscaled), 2)
        ref_batch_0 = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        ref_batch_2 = ref_batch_0.copy()
        for i in range(3):
            keep = target == i
            ref_batch_2[:, i] += np.sum(impact[:, keep], axis=1)
        np.testing.assert_allclose(unscaled[0][0], ref_batch_0)
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_batch_0.shape[0]))
        np.testing.assert_allclose(unscaled[1][0], ref_batch_2)
        np.testing.assert_allclose(unscaled[1][1], np.sum(ref_batch_2, axis=1))

    # method _compute_update() already tested by the above

    def test_evaluate_1(self):
        manager = ClassifierSetManager(target=TARGET)
        scaled = manager.evaluate(features=REFERENCE, num_batches=None)
        # check default values for set manager with no data
        self.assertEqual(len(scaled), 1)
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_2(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO_NO_PROTOTYPES)  # batch with no prototypes has no impact on the model
        scaled = manager.evaluate(features=REFERENCE, num_batches=0)
        self.assertEqual(len(scaled), 1)
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_3(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        scaled = manager.evaluate(features=REFERENCE, num_batches=np.array([0]))  # evaluate marginals only
        self.assertEqual(len(scaled), 1)
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_4(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        scaled = manager.evaluate(features=REFERENCE, num_batches=None)
        self.assertEqual(len(scaled), 1)
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, _ = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch],
            num_batches=1,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        for i in range(3):
            keep = target == i
            ref_scaled[:, i] += np.sum(impact[:, keep], axis=1)
        ref_scaled = (ref_scaled.transpose() / np.sum(ref_scaled, axis=1)).transpose()
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_5(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        scaled = manager.evaluate(features=REFERENCE, num_batches=np.array([0, 2]))
        self.assertEqual(len(scaled), 2)
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        impact, target, _ = ClassifierSetManager._compute_impact(
            features=REFERENCE,
            batches=[batch, batch],
            num_batches=2,
            meta={"num_features": PROTOTYPES.shape[1]}
        )
        ref_batch_0 = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        ref_batch_2 = ref_batch_0.copy()
        for i in range(3):
            keep = target == i
            ref_batch_2[:, i] += np.sum(impact[:, keep], axis=1)
        ref_batch_2 = (ref_batch_2.transpose() / np.sum(ref_batch_2, axis=1)).transpose()
        np.testing.assert_allclose(scaled[0], ref_batch_0)
        np.testing.assert_allclose(scaled[1], ref_batch_2)

    def test_get_feature_weights_1(self):
        manager = ClassifierSetManager(target=TARGET)
        result = manager.get_feature_weights()  # check default values for set manager with no data
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result["weight_matrix"], np.zeros((0, 0)))
        np.testing.assert_allclose(result["feature_index"], np.zeros(0, dtype=int))

    def test_get_feature_weights_2(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO_NO_FEATURES)  # check behavior in case of no active features
        result = manager.get_feature_weights()
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result["weight_matrix"], np.zeros((1, 0)))
        np.testing.assert_allclose(result["feature_index"], np.zeros(0, dtype=int))

    def test_get_feature_weights_3(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        result = manager.get_feature_weights()
        ref_feature_index = np.where(FEATURE_WEIGHTS > 0.0)[0]
        order = np.argsort(FEATURE_WEIGHTS[ref_feature_index])[-1::-1]
        ref_weight_matrix = FEATURE_WEIGHTS[ref_feature_index][order][np.newaxis, :]
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result["weight_matrix"], ref_weight_matrix)
        np.testing.assert_allclose(result["feature_index"], ref_feature_index[order])

    def test_get_feature_weights_4(self):
        manager = ClassifierSetManager(target=TARGET)
        manager.add_batch(BATCH_INFO)
        new_weights = np.arange(FEATURE_WEIGHTS.shape[0], dtype=float)
        manager.add_batch({
            "prototypes": PROTOTYPES,
            "target": TARGET,
            "feature_weights": new_weights,
            "prototype_weights": PROTOTYPE_WEIGHTS,
            "sample_index": SAMPLE_INDEX
        })  # the new weights mean all features are active now
        result = manager.get_feature_weights()
        order = np.argsort(FEATURE_WEIGHTS)[-1::-1]
        ref_weight_matrix = np.vstack([FEATURE_WEIGHTS[order], new_weights[order]])
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result["weight_matrix"], ref_weight_matrix)
        np.testing.assert_allclose(result["feature_index"], order)
