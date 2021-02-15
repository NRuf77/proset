"""Unit tests for code in shared.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

import proset.shared as shared
from test.test_set_manager import REFERENCE, PROTOTYPES, FEATURE_WEIGHTS


class TestShared(TestCase):
    """Unit tests for functions in shared.py.
    """

    def test_check_classifier_target_fail_1(self):
        message = ""
        try:
            shared.check_classifier_target(np.array([0, 2]))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "The classes must be encoded as integers from 0 to K - 1 and each class must be present."
        )

    def test_check_classifier_target_1(self):
        result = shared.check_classifier_target(np.array([0, 1, 1, 0, 2]))
        np.testing.assert_allclose(result, np.array([2, 2, 1]))

    def test_find_changes_1(self):
        result = shared.find_changes(np.array([0, 0, 1, 1, 1, 2]))
        np.testing.assert_allclose(result, np.array([0, 2, 5]))

    def test_quick_compute_impact_1(self):
        scaled_reference = REFERENCE * FEATURE_WEIGHTS
        scaled_prototypes = PROTOTYPES * FEATURE_WEIGHTS
        similarity = shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=scaled_prototypes,
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=np.sum(scaled_prototypes ** 2.0, axis=1)
        )
        reference_similarity = np.zeros((REFERENCE.shape[0], PROTOTYPES.shape[0]), dtype=float)
        for i in range(REFERENCE.shape[0]):
            for j in range(PROTOTYPES.shape[0]):
                reference_similarity[i, j] = np.exp(
                    -0.5 * np.sum(((REFERENCE[i] - PROTOTYPES[j]) * FEATURE_WEIGHTS) ** 2.0)
                )
        np.testing.assert_allclose(similarity, reference_similarity)
