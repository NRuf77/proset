"""Unit tests for code in shared.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

import proset.shared as shared
from test.test_set_manager import REFERENCE, PROTOTYPES, FEATURE_WEIGHTS  # pylint: disable=wrong-import-order


FEATURE_NAMES = ["feature_1", "feature_2"]


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestShared(TestCase):
    """Unit tests for functions in shared.py.
    """

    def test_check_classifier_target_fail_1(self):
        message = ""
        try:
            shared.check_classifier_target(np.array([[0, 1]]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must be a 1D array.")

    def test_check_classifier_target_fail_2(self):
        message = ""
        try:
            shared.check_classifier_target(np.array([0.0, 1.0]))
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter target must be an integer array.")

    def test_check_classifier_target_fail_3(self):
        message = ""
        try:
            shared.check_classifier_target(np.array([0, 2]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter target must encode classes as integers from 0 to K - 1 and every class must be present."
        )

    @staticmethod
    def test_check_classifier_target_1():
        result = shared.check_classifier_target(np.array([0, 1, 1, 0, 2]))
        np.testing.assert_allclose(result, np.array([2, 2, 1]))

    @staticmethod
    def test_find_changes_1():
        result = shared.find_changes(np.array([0, 0, 1, 1, 1, 2]))
        np.testing.assert_allclose(result, np.array([0, 2, 5]))

    def test_quick_compute_similarity_1(self):
        scaled_reference = REFERENCE * FEATURE_WEIGHTS
        scaled_prototypes = PROTOTYPES * FEATURE_WEIGHTS
        similarity = shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=scaled_prototypes,
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=np.sum(scaled_prototypes ** 2.0, axis=1)
        )
        self.assertEqual(similarity.dtype, np.float32)
        self.assertTrue(similarity.flags["F_CONTIGUOUS"])
        reference_similarity = np.zeros((REFERENCE.shape[0], PROTOTYPES.shape[0]), dtype=float)
        for i in range(REFERENCE.shape[0]):
            for j in range(PROTOTYPES.shape[0]):
                reference_similarity[i, j] = np.exp(
                    -0.5 * np.sum(((REFERENCE[i] - PROTOTYPES[j]) * FEATURE_WEIGHTS) ** 2.0)
                )
        np.testing.assert_allclose(similarity, reference_similarity, atol=1e-6)

    def test_check_feature_names_fail_1(self):
        message = ""
        try:
            shared.check_feature_names(num_features=1.0, feature_names=None, active_features=None)
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_features must be integer.")

    def test_check_feature_names_fail_2(self):
        message = ""
        try:
            shared.check_feature_names(num_features=0, feature_names=None, active_features=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_features must be positive.")

    def test_check_feature_names_fail_3(self):
        message = ""
        try:
            shared.check_feature_names(num_features=1, feature_names=FEATURE_NAMES, active_features=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter feature_names must have one element per feature if not None.")

    def test_check_feature_names_fail_4(self):
        message = ""
        try:
            shared.check_feature_names(num_features=2, feature_names=None, active_features=np.array([[0, 1]]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter active_features must be a 1D array.")

    def test_check_feature_names_fail_5(self):
        message = ""
        try:
            shared.check_feature_names(num_features=2, feature_names=None, active_features=np.array([0.0, 1.0]))
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter active_features must be an integer array.")

    def test_check_feature_names_fail_6(self):
        message = ""
        try:
            shared.check_feature_names(num_features=2, feature_names=None, active_features=np.array([-1, 0]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message,
            "Parameter active_features must contain non-negative numbers less than the total number of features."
        )

    def test_check_feature_names_fail_7(self):
        message = ""
        try:
            shared.check_feature_names(num_features=2, feature_names=None, active_features=np.array([0, 2]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message,
            "Parameter active_features must contain non-negative numbers less than the total number of features."
        )

    def test_check_feature_names_1(self):
        result = shared.check_feature_names(num_features=2, feature_names=FEATURE_NAMES, active_features=None)
        self.assertEqual(result, FEATURE_NAMES)
        self.assertFalse(result is FEATURE_NAMES)  # ensure result is a copy and not a reference to the original input

    def test_check_feature_names_2(self):
        result = shared.check_feature_names(num_features=2, feature_names=None, active_features=None)
        self.assertEqual(result, ["X0", "X1"])

    def test_check_feature_names_3(self):
        result = shared.check_feature_names(num_features=2, feature_names=FEATURE_NAMES, active_features=np.array([1]))
        self.assertEqual(result, [FEATURE_NAMES[1]])

    def test_check_feature_names_4(self):
        result = shared.check_feature_names(num_features=2, feature_names=None, active_features=np.array([1]))
        self.assertEqual(result, ["X1"])

    def test_check_scale_offset_fail_1(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=2.0, scale=None, offset=None)
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_features must be integer.")

    def test_check_scale_offset_fail_2(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=0, scale=None, offset=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_features must be positive.")

    def test_check_scale_offset_fail_3(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=2, scale=np.array([[0.5, 2.0]]), offset=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter scale must be a 1D array.")

    def test_check_scale_offset_fail_4(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=3, scale=np.array([0.5, 2.0]), offset=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter scale must have one element per feature.")

    def test_check_scale_offset_fail_5(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=2, scale=np.array([0.0, 2.0]), offset=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter scale must have strictly positive elements.")

    def test_check_scale_offset_fail_6(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=2, scale=np.array([0.5, 2.0]), offset=np.array([[-1.0, 1.0]]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter offset must be a 1D array.")

    def test_check_scale_offset_fail_7(self):
        message = ""
        try:
            shared.check_scale_offset(num_features=3, scale=np.array([0.5, 2.0, 1.0]), offset=np.array([-1.0, 1.0]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter offset must have one element per feature.")

    def test_check_scale_offset_1(self):
        scale_in = np.array([0.5, 2.0])
        offset_in = np.array([-1.0, 1.0])
        scale_out, offset_out = shared.check_scale_offset(num_features=2, scale=scale_in, offset=offset_in)
        shared.check_float_array(x=scale_out, name="scale_out")
        np.testing.assert_array_equal(scale_out, scale_in)
        self.assertFalse(scale_out is scale_in)  # ensure result is a copy and not a reference to the original input
        shared.check_float_array(x=offset_out, name="offset_out")
        np.testing.assert_array_equal(offset_out, offset_in)
        self.assertFalse(offset_out is offset_in)

    @staticmethod
    def test_check_scale_offset_2():
        scale_out, offset_out = shared.check_scale_offset(num_features=2, scale=None, offset=None)
        shared.check_float_array(x=scale_out, name="scale_out")
        np.testing.assert_array_equal(scale_out, np.ones(2, **shared.FLOAT_TYPE))
        shared.check_float_array(x=offset_out, name="offset_out")
        np.testing.assert_array_equal(offset_out, np.zeros(2, **shared.FLOAT_TYPE))

    def test_check_float_array_fail_1(self):
        message = ""
        try:
            shared.check_float_array(x=np.zeros((2, 2), dtype=np.float64, order="F"), name="x")
            # array properties for triggering the check are hard-coded, change them if you update shared.FLOAT_TYPE
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter x must be an array of type float32.")

    def test_check_float_array_fail_2(self):
        message = ""
        try:
            shared.check_float_array(x=np.zeros((2, 2), dtype=np.float32, order="C"), name="x")
            # array properties for triggering the check are hard-coded, change them if you update shared.FLOAT_TYPE
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter x must be a Fortran-contiguous array.")

    def test_check_float_array_fail_3(self):
        message = ""
        try:
            shared.check_float_array(
                x=np.zeros((2, 2), dtype=np.float32, order="F"), name="x", spec={"dtype": np.float32, "order": "C"}
                # array properties for triggering the check are hard-coded, change them if you update shared.FLOAT_TYPE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter x must be a C-contiguous array.")

    @staticmethod
    def test_check_float_array_1():
        shared.check_float_array(x=np.zeros((2, 2), **shared.FLOAT_TYPE), name="x")

    @staticmethod
    def test_stack_first_1():
        result = shared.stack_first([np.array([1, 2, 3]), np.array([4, 5, 6])])
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5, 6]))

    @staticmethod
    def test_stack_first_2():
        result = shared.stack_first([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])])
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
