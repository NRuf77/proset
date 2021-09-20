"""Unit tests for code in the utility submodule.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

import proset.utility.other as other


class TestUtility(TestCase):

    def test_find_best_point_1(self):
        features = np.array([[4], [2], [2], [1]])
        prediction = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        # this constellation gives the same Borda points to samples 1 and 2 but the probability estimate for sample 2
        # has higher entropy and is preferred
        result = other._find_best_point(features=features, prediction=prediction, is_classifier_=True)
        self.assertEqual(result, 2)

    def test__compute_borda_points_1(self):
        metric = np.array([5.0, 2.0, 4.0, 3.0, 2.0, 1.0])
        reference = np.array([0.0, 3.5, 1.0, 2.0, 3.5, 5.0])
        result = other._compute_borda_points(metric)
        np.testing.assert_array_equal(result, reference)
