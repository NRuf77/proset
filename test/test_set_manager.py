"""Unit tests for code in set_manager.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

from proset import shared
from proset.set_manager import ClassifierSetManager


# define common objects for testing
TARGET = np.array([0, 1, 0, 2, 1, 2])
_, MARGINALS = np.unique(TARGET, return_counts=True)
MARGINALS = (MARGINALS / np.sum(MARGINALS)).astype(**shared.FLOAT_TYPE)
PROTOTYPES = np.array([
    [1.0, 0.0, 0.0, 2.7],
    [1.0, 0.0, 0.0, 1.8],
    [0.0, 1.0, 0.0, -3.5],
    [0.0, 1.0, 0.0, -1.9],
    [0.0, 0.0, 1.0, 12.0],
    [0.0, 0.0, 1.0, 8.0]
], **shared.FLOAT_TYPE)
FEATURE_WEIGHTS = np.array([0.5, 0.0, 1.5, 0.1], **shared.FLOAT_TYPE)
PROTOTYPE_WEIGHTS = np.array([1.0, 2.0, 0.0, 2.0, 1.0, 0.5], **shared.FLOAT_TYPE)
SAMPLE_INDEX = np.array([4, 7, 11, 15, 27, 40])
REFERENCE = np.array([
    [1.0, 0.0, 0.0, 3.0],
    [0.0, 1.0, 0.0, -2.0],
    [0.0, 0.0, 1.0, 9.5]
], **shared.FLOAT_TYPE)
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
    "feature_weights": np.zeros_like(FEATURE_WEIGHTS, **shared.FLOAT_TYPE),
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_NO_PROTOTYPES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": FEATURE_WEIGHTS,
    "prototype_weights": np.zeros_like(PROTOTYPE_WEIGHTS, **shared.FLOAT_TYPE),
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_ALL_FEATURES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": np.ones_like(FEATURE_WEIGHTS, **shared.FLOAT_TYPE),
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestClassifierSetManager(TestCase):
    """Unit tests class ClassifierSetManager.

    The tests also cover abstract superclass SetManager.
    """

    def test_init_fail_1(self):
        message = ""
        try:
            # test only one exception to ensure shared.check_classifier_target() is called; other exceptions tested by
            # the unit tests for that function
            ClassifierSetManager(target=np.array([[0, 1]]), weights=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must be a 1D array.")

    def test_init_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        self.assertEqual(manager._target_type, {"dtype": int})
        self.assertEqual(manager.num_batches, 0)
        self.assertEqual(manager.num_features, None)
        shared.check_float_array(x=manager.marginals, name="manager.marginals")
        np.testing.assert_allclose(manager.marginals, MARGINALS)
        self.assertEqual(manager.get_active_features().shape[0], 0)
        self.assertEqual(manager.get_num_prototypes(), 0)

    def test_init_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=np.ones_like(TARGET, **shared.FLOAT_TYPE))
        self.assertEqual(manager._target_type, {"dtype": int})
        self.assertEqual(manager.num_batches, 0)
        self.assertEqual(manager.num_features, None)
        shared.check_float_array(x=manager.marginals, name="manager.marginals")
        np.testing.assert_allclose(manager.marginals, MARGINALS)
        self.assertEqual(manager.get_active_features().shape[0], 0)
        self.assertEqual(manager.get_num_prototypes(), 0)

    # methods _get_baseline_distribution() already tested by test_init_1() above
    # property getters for num_batches, num_features, and marginals already tested by the above and below
    # method get_active_features() tested above and below

    def test_check_num_batches_fail_1(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(
                num_batches=np.array([1, 2]), num_batches_actual=3, permit_array=False
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must not be an array.")

    def test_check_num_batches_fail_2(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(
                num_batches=np.array([[1, 2]]), num_batches_actual=3, permit_array=True
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must be 1D if passing an array.")

    def test_check_num_batches_fail_3(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(
                num_batches=np.array([1.0, 2.0]), num_batches_actual=3, permit_array=True
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must be of integer type if passing an array.")

    def test_check_num_batches_fail_4(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(
                num_batches=np.array([-1, 2]), num_batches_actual=3, permit_array=True
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must not contain negative values if passing an array.")

    def test_check_num_batches_fail_5(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(
                num_batches=np.array([1, 4]), num_batches_actual=3, permit_array=True
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message,
            "Parameter num_batches must not contain values greater than the available number of batches (3) if " +
            "passing an array."
        )

    def test_check_num_batches_fail_6(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(
                num_batches=np.array([1, 1]), num_batches_actual=3, permit_array=True
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter num_batches must contain strictly increasing elements if passing an array."
        )

    def test_check_num_batches_fail_7(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(num_batches=1.0, num_batches_actual=3, permit_array=False)
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must be an integer.")

    def test_check_num_batches_fail_8(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(num_batches=-1, num_batches_actual=3, permit_array=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must not be negative.")

    def test_check_num_batches_fail_9(self):
        message = ""
        try:
            ClassifierSetManager._check_num_batches(num_batches=4, num_batches_actual=3, permit_array=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter num_batches must be less than or equal to the available number of batches (3)."
        )

    def test_check_num_batches_1(self):
        result = ClassifierSetManager._check_num_batches(num_batches=None, num_batches_actual=3, permit_array=False)
        self.assertEqual(result, 3)

    def test_check_num_batches_2(self):
        result = ClassifierSetManager._check_num_batches(num_batches=1, num_batches_actual=3, permit_array=False)
        self.assertEqual(result, 1)

    def test_check_num_batches_3(self):
        num_batches = np.array([1, 2])
        result = ClassifierSetManager._check_num_batches(
            num_batches=num_batches, num_batches_actual=3, permit_array=True
        )
        np.testing.assert_array_equal(result, num_batches)
        self.assertFalse(result is num_batches)  # ensure result is a copy and not a reference to the original input

    # method get_num_prototypes() tested above and below

    def test_add_batch_fail_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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
        manager = ClassifierSetManager(target=TARGET, weights=None)
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
        manager = ClassifierSetManager(target=TARGET, weights=None)
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
        manager = ClassifierSetManager(target=TARGET, weights=None)
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
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch({
            "prototypes": PROTOTYPES,
            "target": TARGET,
            "feature_weights": FEATURE_WEIGHTS,
            "prototype_weights": np.zeros_like(PROTOTYPE_WEIGHTS, **shared.FLOAT_TYPE),
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
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:  # trigger one check from shared.check_float_array() to ensure it is called
            manager.add_batch({
                "prototypes": PROTOTYPES.astype(np.float64),
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter prototypes must be an array of type float32.")

    def test_add_batch_fail_7(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_8(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_9(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_10(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_11(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:  # trigger one check from shared.check_float_array() to ensure it is called
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS.astype(np.float64),
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX
            })
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter feature_weights must be an array of type float32.")

    def test_add_batch_fail_12(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_13(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_14(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:  # trigger one check from shared.check_float_array() to ensure it is called
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS.astype(np.float64),
                "sample_index": SAMPLE_INDEX
            })
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter prototype_weights must be an array of type float32.")

    def test_add_batch_fail_15(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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

    def test_add_batch_fail_16(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:
            manager.add_batch({
                "prototypes": PROTOTYPES,
                "target": TARGET,
                "feature_weights": FEATURE_WEIGHTS,
                "prototype_weights": PROTOTYPE_WEIGHTS,
                "sample_index": SAMPLE_INDEX.astype(float)
            })
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter sample_index must be an integer array.")

    def test_add_batch_fail_17(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
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
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        self.assertEqual(manager.num_batches, 1)
        self.assertEqual(manager.num_features, len(FEATURE_WEIGHTS))
        self.assertEqual(manager.get_active_features().shape[0], np.sum(FEATURE_WEIGHTS > 0.0))
        self.assertEqual(manager.get_num_prototypes(), np.sum(PROTOTYPE_WEIGHTS > 0.0))

    def test_add_batch_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        self.assertEqual(manager.num_batches, 2)
        self.assertEqual(manager.num_features, len(FEATURE_WEIGHTS))
        self.assertEqual(manager.get_active_features().shape[0], np.sum(FEATURE_WEIGHTS > 0.0))
        self.assertEqual(manager.get_num_prototypes(), 2 * np.sum(PROTOTYPE_WEIGHTS > 0.0))
        # the same training sample added to two different batches counts as two prototypes

    def test_add_batch_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO_NO_PROTOTYPES)
        self.assertEqual(manager.num_batches, 1)
        self.assertEqual(manager.num_features, len(FEATURE_WEIGHTS))
        self.assertEqual(manager.get_active_features().shape[0], 0)
        # batch with no prototypes is counted but does not contribute anything
        self.assertEqual(manager.get_num_prototypes(), 0)

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
        unique_weights = np.zeros_like(unique_target, **shared.FLOAT_TYPE)
        unique_index = np.zeros_like(unique_target)
        for i, value in enumerate(unique_target):
            unique_weights[i] = np.sum(PROTOTYPE_WEIGHTS[TARGET == value])
            unique_index[i] = np.min(SAMPLE_INDEX[TARGET == value])
        shared.check_float_array(x=result["scaled_prototypes"], name="result['scaled_prototypes']")
        np.testing.assert_allclose(result["scaled_prototypes"], np.zeros((unique_target.shape[0], 0)))
        shared.check_float_array(x=result["ssq_prototypes"], name="result['ssq_prototypes']")
        np.testing.assert_allclose(result["ssq_prototypes"], np.zeros(unique_target.shape[0]))
        np.testing.assert_allclose(result["target"], unique_target)
        shared.check_float_array(x=result["feature_weights"], name="result['feature_weights']")
        np.testing.assert_allclose(result["feature_weights"], np.zeros(0))
        shared.check_float_array(x=result["prototype_weights"], name="result['prototype_weights']")
        np.testing.assert_allclose(result["prototype_weights"], unique_weights)
        np.testing.assert_allclose(result["sample_index"], unique_index)

    def test_process_batch_3(self):
        result = ClassifierSetManager._process_batch(BATCH_INFO)
        self.assertEqual(len(result), 7)
        active_features = np.nonzero(FEATURE_WEIGHTS > 0.0)[0]
        np.testing.assert_allclose(result["active_features"], active_features)
        active_prototypes = np.nonzero(PROTOTYPE_WEIGHTS > 0.0)[0]
        shared.check_float_array(x=result["scaled_prototypes"], name="result['scaled_prototypes']")
        scaled_prototypes = PROTOTYPES[:, active_features][active_prototypes, :] * FEATURE_WEIGHTS[active_features]
        np.testing.assert_allclose(result["scaled_prototypes"], scaled_prototypes)
        shared.check_float_array(x=result["ssq_prototypes"], name="result['ssq_prototypes']")
        np.testing.assert_allclose(result["ssq_prototypes"], np.sum(scaled_prototypes ** 2.0, axis=1))
        np.testing.assert_allclose(result["target"], TARGET[active_prototypes])
        shared.check_float_array(x=result["feature_weights"], name="result['feature_weights']")
        np.testing.assert_allclose(result["feature_weights"], FEATURE_WEIGHTS[active_features])
        shared.check_float_array(x=result["prototype_weights"], name="result['prototype_weights']")
        np.testing.assert_allclose(result["prototype_weights"], PROTOTYPE_WEIGHTS[active_prototypes])
        np.testing.assert_allclose(result["sample_index"], SAMPLE_INDEX[active_prototypes])

    def test_process_batch_4(self):
        result = ClassifierSetManager._process_batch({
            "prototypes": np.ones((TARGET.shape[0], 1), **shared.FLOAT_TYPE),
            # prototype values are redundant so should be merged by _process_batch()
            "target": TARGET,
            "feature_weights": np.ones(1, **shared.FLOAT_TYPE),
            "prototype_weights": PROTOTYPE_WEIGHTS,
            "sample_index": SAMPLE_INDEX
        })
        self.assertEqual(len(result), 7)
        np.testing.assert_allclose(result["active_features"], np.array([0]))
        shared.check_float_array(x=result["scaled_prototypes"], name="result['scaled_prototypes']")
        np.testing.assert_allclose(result["scaled_prototypes"], np.ones((3, 1)))
        shared.check_float_array(x=result["ssq_prototypes"], name="result['scaled_prototypes']")
        np.testing.assert_allclose(result["ssq_prototypes"], np.ones(3))
        reduced_target = np.unique(TARGET)
        np.testing.assert_allclose(result["target"], reduced_target)
        shared.check_float_array(x=result["feature_weights"], name="result['feature_weights']")
        np.testing.assert_allclose(result["feature_weights"], np.ones(1))
        merged_weights = np.array([np.sum(PROTOTYPE_WEIGHTS[TARGET == i]) for i in reduced_target])
        shared.check_float_array(x=result["prototype_weights"], name="result['prototype_weights']")
        np.testing.assert_allclose(result["prototype_weights"], merged_weights)
        reduced_index = np.array([np.min(SAMPLE_INDEX[TARGET == i]) for i in reduced_target])
        np.testing.assert_allclose(result["sample_index"], reduced_index)

    def test_evaluate_unscaled_fail_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:
            manager.evaluate_unscaled(features=REFERENCE[:, 0], num_batches=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features must be a 2D array.")

    def test_evaluate_unscaled_fail_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        message = ""
        try:
            manager.evaluate_unscaled(features=np.zeros((3, 3), **shared.FLOAT_TYPE), num_batches=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features has 3 columns but {} are expected.".format(PROTOTYPES.shape[1]))

    def test_evaluate_unscaled_fail_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        message = ""
        try:  # trigger one check from shared.check_float_array() to ensure it is called
            manager.evaluate_unscaled(features=REFERENCE.astype(np.float64), num_batches=None)
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features must be an array of type float32.")

    def test_evaluate_unscaled_fail_4(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:  # trigger one check from _check_num_batches() to ensure it is called
            manager.evaluate_unscaled(features=REFERENCE, num_batches=np.array([-1]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batches must not contain negative values if passing an array.")

    def test_evaluate_unscaled_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        unscaled = manager.evaluate_unscaled(features=REFERENCE, num_batches=None)
        # no batches means marginal distribution is returned
        self.assertEqual(len(unscaled), 1)
        shared.check_float_array(x=unscaled[0][0], name="unscaled[0][0]")
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        shared.check_float_array(x=unscaled[0][1], name="unscaled[0][1]")
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_evaluate_unscaled_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO_NO_PROTOTYPES)  # batch with no prototypes has no impact on the model
        unscaled = manager.evaluate_unscaled(features=REFERENCE, num_batches=0)
        self.assertEqual(len(unscaled), 1)
        shared.check_float_array(x=unscaled[0][0], name="unscaled[0][0]")
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        shared.check_float_array(x=unscaled[0][1], name="unscaled[0][1]")
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_evaluate_unscaled_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        unscaled = manager.evaluate_unscaled(REFERENCE, num_batches=np.array([0]))  # evaluate marginals only
        self.assertEqual(len(unscaled), 1)
        shared.check_float_array(x=unscaled[0][0], name="unscaled[0][0]")
        ref_unscaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        shared.check_float_array(x=unscaled[0][1], name="unscaled[0][1]")
        np.testing.assert_allclose(unscaled[0][1], np.ones(ref_unscaled.shape[0]))

    def test_evaluate_unscaled_4(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        unscaled = manager.evaluate_unscaled(REFERENCE, num_batches=None)
        self.assertEqual(len(unscaled), 1)
        shared.check_float_array(x=unscaled[0][0], name="unscaled[0][0]")
        meta = {"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        base_unscaled, base_scale = ClassifierSetManager._get_baseline(num_samples=REFERENCE.shape[0], meta=meta)
        ref_unscaled, ref_scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=ClassifierSetManager._process_batch(BATCH_INFO), meta=meta
        )
        ref_unscaled += base_unscaled
        ref_scale += base_scale
        np.testing.assert_allclose(unscaled[0][0], ref_unscaled)
        shared.check_float_array(x=unscaled[0][1], name="unscaled[0][1]")
        np.testing.assert_allclose(unscaled[0][1], ref_scale)

    def test_evaluate_unscaled_5(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        unscaled = manager.evaluate_unscaled(REFERENCE, num_batches=np.array([0, 2]))
        self.assertEqual(len(unscaled), 2)
        shared.check_float_array(x=unscaled[0][0], name="unscaled[0][0]")
        meta = {"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        base_unscaled, base_scale = ClassifierSetManager._get_baseline(num_samples=REFERENCE.shape[0], meta=meta)
        ref_unscaled, ref_scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=ClassifierSetManager._process_batch(BATCH_INFO), meta=meta
        )
        ref_unscaled = base_unscaled + 2.0 * ref_unscaled
        ref_scale = base_scale + 2.0 * ref_scale
        np.testing.assert_allclose(unscaled[0][0], base_unscaled)
        shared.check_float_array(x=unscaled[0][1], name="unscaled[0][1]")
        np.testing.assert_allclose(unscaled[0][1], base_scale)
        shared.check_float_array(x=unscaled[1][0], name="unscaled[1][0]")
        np.testing.assert_allclose(unscaled[1][0], ref_unscaled, atol=1e-6)
        shared.check_float_array(x=unscaled[1][1], name="unscaled[1][1]")
        np.testing.assert_allclose(unscaled[1][1], ref_scale, atol=1e-6)

    # method _check_evaluate_input() already tested by the above

    @staticmethod
    def test_get_sample_ranges_1():
        result = ClassifierSetManager._get_sample_ranges(5000)
        np.testing.assert_array_equal(result, np.array([0, 5000]))

    @staticmethod
    def test_get_sample_ranges_2():
        result = ClassifierSetManager._get_sample_ranges(10001)
        np.testing.assert_array_equal(result, np.array([0, 5001, 10001]))

    @staticmethod
    def test_get_baseline_1():
        unscaled, scale = ClassifierSetManager._get_baseline(num_samples=10, meta={"marginals": MARGINALS})
        shared.check_float_array(x=unscaled, name="unscaled")
        np.testing.assert_array_equal(unscaled, np.vstack([MARGINALS] * 10))
        shared.check_float_array(x=scale, name="scale")
        np.testing.assert_array_equal(scale, np.ones(10, **shared.FLOAT_TYPE))

    @staticmethod
    def test_get_batch_contribution_1():
        batch = ClassifierSetManager._process_batch(BATCH_INFO)
        unscaled, scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=batch, meta={"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        )
        shared.check_float_array(x=unscaled, name="unscaled")
        scaled_reference = REFERENCE[:, batch["active_features"]] * batch["feature_weights"]
        impact = shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=batch["scaled_prototypes"],
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=batch["ssq_prototypes"]
        ) * batch["prototype_weights"]
        reference = np.zeros((REFERENCE.shape[0], MARGINALS.shape[0]), **shared.FLOAT_TYPE)
        for i, label in enumerate(batch["target"]):
            reference[:, label] += impact[:, i]
        np.testing.assert_allclose(unscaled, reference)
        shared.check_float_array(x=scale, name="scale")
        np.testing.assert_allclose(scale, np.sum(reference, axis=1))

    @staticmethod
    def test_get_batch_contribution_2():
        batch = ClassifierSetManager._process_batch(BATCH_INFO_NO_FEATURES)
        unscaled, scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=batch, meta={"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        )
        shared.check_float_array(x=unscaled, name="unscaled")
        scaled_reference = REFERENCE[:, batch["active_features"]] * batch["feature_weights"]
        impact = shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=batch["scaled_prototypes"],
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=batch["ssq_prototypes"]
        ) * batch["prototype_weights"]
        reference = np.zeros((REFERENCE.shape[0], MARGINALS.shape[0]), **shared.FLOAT_TYPE)
        for i, label in enumerate(batch["target"]):
            reference[:, label] += impact[:, i]
        np.testing.assert_allclose(unscaled, reference)
        shared.check_float_array(x=scale, name="scale")
        np.testing.assert_allclose(scale, np.sum(reference, axis=1))

    @staticmethod
    def test_get_batch_contribution_3():
        batch = ClassifierSetManager._process_batch(BATCH_INFO_ALL_FEATURES)
        unscaled, scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=batch, meta={"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        )
        shared.check_float_array(x=unscaled, name="unscaled")
        scaled_reference = REFERENCE[:, batch["active_features"]] * batch["feature_weights"]
        impact = shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=batch["scaled_prototypes"],
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=batch["ssq_prototypes"]
        ) * batch["prototype_weights"]
        reference = np.zeros((REFERENCE.shape[0], MARGINALS.shape[0]), **shared.FLOAT_TYPE)
        for i, label in enumerate(batch["target"]):
            reference[:, label] += impact[:, i]
        np.testing.assert_allclose(unscaled, reference)
        shared.check_float_array(x=scale, name="scale")
        np.testing.assert_allclose(scale, np.sum(reference, axis=1))

    def test_evaluate_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        scaled = manager.evaluate(features=REFERENCE, num_batches=None, compute_familiarity=False)
        # check default values for set manager with no data
        self.assertEqual(len(scaled), 1)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        scaled, familiarity = manager.evaluate(features=REFERENCE, num_batches=None, compute_familiarity=True)
        # check default values for set manager with no data
        self.assertEqual(len(scaled), 1)
        self.assertEqual(len(familiarity), 1)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)
        shared.check_float_array(x=familiarity[0], name="familiarity[0]")
        np.testing.assert_allclose(familiarity[0], np.zeros_like(familiarity[0]))

    def test_evaluate_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO_NO_PROTOTYPES)  # batch with no prototypes has no impact on the model
        scaled = manager.evaluate(features=REFERENCE, num_batches=0, compute_familiarity=False)
        self.assertEqual(len(scaled), 1)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_4(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        scaled = manager.evaluate(features=REFERENCE, num_batches=np.array([0]), compute_familiarity=False)
        # evaluate marginals only
        self.assertEqual(len(scaled), 1)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        ref_scaled = np.tile(MARGINALS, (REFERENCE.shape[0], 1))
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_5(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        scaled = manager.evaluate(features=REFERENCE, num_batches=None, compute_familiarity=False)
        self.assertEqual(len(scaled), 1)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        meta = {"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        base_unscaled, base_scale = ClassifierSetManager._get_baseline(num_samples=REFERENCE.shape[0], meta=meta)
        ref_unscaled, ref_scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=ClassifierSetManager._process_batch(BATCH_INFO), meta=meta
        )
        ref_unscaled += base_unscaled
        ref_scale += base_scale
        ref_scaled = (ref_unscaled.transpose() / ref_scale).transpose()
        np.testing.assert_allclose(scaled[0], ref_scaled)

    def test_evaluate_6(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        scaled, familiarity = manager.evaluate(features=REFERENCE, num_batches=None, compute_familiarity=True)
        self.assertEqual(len(scaled), 1)
        self.assertEqual(len(familiarity), 1)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        meta = {"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        base_unscaled, base_scale = ClassifierSetManager._get_baseline(num_samples=REFERENCE.shape[0], meta=meta)
        ref_unscaled, ref_scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=ClassifierSetManager._process_batch(BATCH_INFO), meta=meta
        )
        ref_unscaled += base_unscaled
        ref_scale += base_scale
        ref_scaled = (ref_unscaled.transpose() / ref_scale).transpose()
        np.testing.assert_allclose(scaled[0], ref_scaled)
        shared.check_float_array(x=familiarity[0], name="familiarity[0]")
        np.testing.assert_allclose(familiarity[0], ref_scale - 1.0)

    def test_evaluate_7(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        scaled = manager.evaluate(features=REFERENCE, num_batches=np.array([0, 2]), compute_familiarity=False)
        self.assertEqual(len(scaled), 2)
        shared.check_float_array(x=scaled[0], name="scaled[0]")
        meta = {"num_features": REFERENCE.shape[1], "marginals": MARGINALS}
        base_unscaled, base_scale = ClassifierSetManager._get_baseline(num_samples=REFERENCE.shape[0], meta=meta)
        ref_unscaled, ref_scale = ClassifierSetManager._get_batch_contribution(
            features=REFERENCE, batch=ClassifierSetManager._process_batch(BATCH_INFO), meta=meta
        )
        ref_unscaled = base_unscaled + 2.0 * ref_unscaled
        ref_scale = base_scale + 2.0 * ref_scale
        ref_scaled = (ref_unscaled.transpose() / ref_scale).transpose()
        np.testing.assert_allclose(scaled[0], base_unscaled)  # baseline scale is 1.0
        shared.check_float_array(x=scaled[1], name="scaled[1]")
        np.testing.assert_allclose(scaled[1], ref_scaled, atol=1e-6)

    def test_get_feature_weights_fail_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:
            # test only one exception from _check_num_batches() to ensure it is called; other exceptions tested by the
            # unit tests for _check_num_batches()
            manager.get_feature_weights(num_batches=1)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter num_batches must be less than or equal to the available number of batches (0)."
        )

    def test_get_feature_weights_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        result = manager.get_feature_weights()  # check default values for set manager with no data
        self.assertEqual(len(result), 2)
        shared.check_float_array(x=result["weight_matrix"], name="result['weight_matrix']")
        np.testing.assert_allclose(result["weight_matrix"], np.zeros((0, 0)))
        np.testing.assert_allclose(result["feature_index"], np.zeros(0, dtype=int))

    def test_get_feature_weights_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO_NO_FEATURES)  # check behavior in case of no active features
        result = manager.get_feature_weights()
        self.assertEqual(len(result), 2)
        shared.check_float_array(x=result["weight_matrix"], name="result['weight_matrix']")
        np.testing.assert_allclose(result["weight_matrix"], np.zeros((1, 0)))
        np.testing.assert_allclose(result["feature_index"], np.zeros(0, dtype=int))

    def test_get_feature_weights_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        result = manager.get_feature_weights()
        ref_feature_index = np.nonzero(FEATURE_WEIGHTS > 0.0)[0]
        order = np.argsort(FEATURE_WEIGHTS[ref_feature_index])[-1::-1]
        ref_weight_matrix = FEATURE_WEIGHTS[ref_feature_index][order][np.newaxis, :]
        self.assertEqual(len(result), 2)
        shared.check_float_array(x=result["weight_matrix"], name="result['weight_matrix']")
        np.testing.assert_allclose(result["weight_matrix"], ref_weight_matrix)
        np.testing.assert_allclose(result["feature_index"], ref_feature_index[order])

    def test_get_feature_weights_4(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        new_weights = np.arange(FEATURE_WEIGHTS.shape[0]).astype(**shared.FLOAT_TYPE)
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
        shared.check_float_array(x=result["weight_matrix"], name="result['weight_matrix']")
        np.testing.assert_allclose(result["weight_matrix"], ref_weight_matrix)
        np.testing.assert_allclose(result["feature_index"], order)

    def test_get_feature_weights_5(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        manager.add_batch(BATCH_INFO)
        result = manager.get_feature_weights(num_batches=1)
        ref_feature_index = np.nonzero(FEATURE_WEIGHTS > 0.0)[0]
        order = np.argsort(FEATURE_WEIGHTS[ref_feature_index])[-1::-1]
        ref_weight_matrix = FEATURE_WEIGHTS[ref_feature_index][order][np.newaxis, :]
        self.assertEqual(len(result), 2)
        shared.check_float_array(x=result["weight_matrix"], name="result['weight_matrix']")
        np.testing.assert_allclose(result["weight_matrix"], ref_weight_matrix)
        np.testing.assert_allclose(result["feature_index"], ref_feature_index[order])

    def test_get_batches_fail_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:
            # test only one exception from _check_num_batches() to ensure it is called; other exceptions tested by the
            # unit tests for _check_num_batches()
            manager.get_batches(features=None, num_batches=1)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter num_batches must be less than or equal to the available number of batches (0)."
        )

    def test_get_batches_fail_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:
            # test only one exception from _check_evaluate_input() to ensure it is called; other exceptions tested by
            # the unit tests for _check_evaluate_input()
            manager.get_batches(features=REFERENCE[0:1, :].astype(**shared.FLOAT_TYPE), num_batches=1)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter num_batches must be less than or equal to the available number of batches (0)."
        )

    def test_get_batches_fail_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        message = ""
        try:
            manager.get_batches(features=REFERENCE)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter features must have exactly one row.")

    def test_get_batches_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        batches = manager.get_batches()
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(np.sort(list(batches[0].keys())), np.array([
            "active_features", "feature_weights", "prototype_weights", "prototypes", "sample_index", "target"
        ]))
        active_features = np.nonzero(BATCH_INFO["feature_weights"])[0]
        active_prototypes = np.nonzero(BATCH_INFO["prototype_weights"])[0]
        np.testing.assert_array_equal(batches[0]["active_features"], active_features)
        shared.check_float_array(x=batches[0]["prototypes"], name="batches[0]['prototypes']")
        np.testing.assert_array_almost_equal(
            batches[0]["prototypes"], BATCH_INFO["prototypes"][active_prototypes][:, active_features]
        )
        np.testing.assert_array_equal(batches[0]["target"], BATCH_INFO["target"][active_prototypes])
        shared.check_float_array(x=batches[0]["feature_weights"], name="batches[0]['feature_weights']")
        np.testing.assert_array_equal(batches[0]["feature_weights"], BATCH_INFO["feature_weights"][active_features])
        shared.check_float_array(x=batches[0]["prototype_weights"], name="batches[0]['prototype_weights']")
        np.testing.assert_array_equal(
            batches[0]["prototype_weights"], BATCH_INFO["prototype_weights"][active_prototypes]
        )
        np.testing.assert_array_equal(batches[0]["sample_index"], BATCH_INFO["sample_index"][active_prototypes])

    def test_get_batches_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO_NO_FEATURES)
        batches = manager.get_batches()
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(np.sort(list(batches[0].keys())), np.array([
            "active_features", "feature_weights", "prototype_weights", "prototypes", "sample_index", "target"
        ]))
        self.assertEqual(batches[0]["active_features"].shape[0], 0)
        shared.check_float_array(x=batches[0]["prototypes"], name="batches[0]['prototypes']")
        self.assertEqual(batches[0]["prototypes"].shape, (np.unique(TARGET).shape[0], 0))
        # note that the prototypes are merged to one representative per class present in TARGET

        # correctness of computations is checked above, this test is only to ensure get_batches() can handle batches
        # with no prototypes

    def test_get_batches_3(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO_NO_PROTOTYPES)
        batches = manager.get_batches()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], None)

    def test_get_batches_4(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        batches = manager.get_batches(features=REFERENCE[:1].astype(**shared.FLOAT_TYPE))
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(np.sort(list(batches[0].keys())), np.array([
            "active_features", "feature_weights", "prototype_weights", "prototypes", "sample_index", "similarities",
            "target"
        ]))
        active_features = np.nonzero(BATCH_INFO["feature_weights"])[0]
        active_prototypes = np.nonzero(BATCH_INFO["prototype_weights"])[0]
        np.testing.assert_array_equal(batches[0]["active_features"], active_features)
        shared.check_float_array(x=batches[0]["prototypes"], name="batches[0]['prototypes']")
        np.testing.assert_array_almost_equal(
            batches[0]["prototypes"], BATCH_INFO["prototypes"][active_prototypes][:, active_features]
        )
        np.testing.assert_array_equal(batches[0]["target"], BATCH_INFO["target"][active_prototypes])
        shared.check_float_array(x=batches[0]["feature_weights"], name="batches[0]['feature_weights']")
        np.testing.assert_array_equal(batches[0]["feature_weights"], BATCH_INFO["feature_weights"][active_features])
        shared.check_float_array(x=batches[0]["prototype_weights"], name="batches[0]['prototype_weights']")
        np.testing.assert_array_equal(
            batches[0]["prototype_weights"], BATCH_INFO["prototype_weights"][active_prototypes]
        )
        np.testing.assert_array_equal(batches[0]["sample_index"], BATCH_INFO["sample_index"][active_prototypes])
        active_features = FEATURE_WEIGHTS > 0.0
        reference = np.exp(-0.5 * (
                (PROTOTYPES[PROTOTYPE_WEIGHTS > 0.0][:, active_features] - REFERENCE[0, active_features])
                * FEATURE_WEIGHTS[active_features]
        ) ** 2.0)
        np.testing.assert_allclose(batches[0]["similarities"], reference)

    # methods _check_get_batches_input() and _compute_feature_similarities() already tested by the above

    def test_shrink_1(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        result = manager.shrink()  # check that shrinking a set manager with no content breaks nothing
        self.assertEqual(manager.num_features, None)
        np.testing.assert_array_equal(result, np.zeros(0, dtype=int))

    def test_shrink_2(self):
        manager = ClassifierSetManager(target=TARGET, weights=None)
        manager.add_batch(BATCH_INFO)
        result = manager.shrink()
        active_features = np.nonzero(FEATURE_WEIGHTS)[0]
        self.assertEqual(manager.num_features, len(active_features))
        np.testing.assert_array_equal(result, active_features)
        np.testing.assert_array_equal(manager._batches[0]["active_features"], np.arange(active_features.shape[0]))
