"""Unit tests for code in model.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

from sklearn.utils.estimator_checks import check_estimator

from proset import ClassifierModel


class TestClassifierModel(TestCase):

    def test_estimator(self):
        """Run sklearn estimator test.
        """
        check_estimator(ClassifierModel())

    # TODO: implement own interface tests
