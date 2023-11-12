"""Unit tests for code in the utility.other and utility.writer submodules.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np

from proset import ClassifierModel
from proset.shared import check_feature_names
from proset.utility import other, write
from test.test_np_objective import FEATURES, TARGET  # pylint: disable=wrong-import-order


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestUtilityExFit(TestCase):
    """Unit tests for selected helper functions in proset.utility.other and proset.utility.writer.
    """

    # no tests for other.print_hyperparameter_report() and other.print_feature_report() which provide console output
    # only

    def test_choose_reference_point_fail_1(self):
        model = ClassifierModel(n_iter=0)
        model.fit(X=FEATURES, y=TARGET)
        message = ""
        try:
            # test only one exception raised by shared.check_scale_offset() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            other.choose_reference_point(
                features=FEATURES,
                model=model,
                scale=np.ones(FEATURES.shape[0] + 1),
                offset=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter scale must have one element per feature.")

    def test_choose_reference_point_1(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        active_features = model.set_manager_.get_active_features()
        result = other.choose_reference_point(features=FEATURES, model=model, scale=None, offset=None)
        self.assertEqual(len(result), 6)
        ix = result["index"]
        self.assertTrue(ix in range(FEATURES.shape[0]))
        np.testing.assert_array_equal(result["features_raw"], FEATURES[ix:(ix + 1), :])
        self.assertFalse(result["features_raw"] is FEATURES[ix:(ix + 1), :])  # ensure submatrix is a copy
        np.testing.assert_array_equal(result["features_processed"], FEATURES[ix:(ix + 1), active_features])
        np.testing.assert_array_equal(result["prediction"], np.squeeze(model.predict_proba(X=FEATURES[ix:(ix + 1)])))
        self.assertEqual(result["num_features"], FEATURES.shape[1])
        np.testing.assert_array_equal(result["active_features"], active_features)

    def test_find_best_point_1(self):
        features = np.zeros((4, 0))
        prediction = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        # this constellation gives the same Borda points to samples 1 and 2 but the probability estimate for sample 2
        # has higher entropy and is preferred
        result = other._find_best_point(features=features, prediction=prediction, is_classifier_=True)
        self.assertEqual(result, 2)

    def test_find_best_point_2(self):
        features = np.array([[4.0], [2.0], [2.0], [1.0]])
        prediction = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.9, 0.1]])
        # this constellation gives the same Borda points to samples 1 and 2 but the probability estimate for sample 2
        # has higher entropy and is preferred
        result = other._find_best_point(features=features, prediction=prediction, is_classifier_=True)
        self.assertEqual(result, 2)

    @staticmethod
    def test_compute_borda_points_1():
        metric = np.array([5.0, 2.0, 4.0, 3.0, 2.0, 1.0])
        reference = np.array([0.0, 3.5, 1.0, 2.0, 3.5, 5.0])
        result = other._compute_borda_points(metric)
        np.testing.assert_array_equal(result, reference)

    # no tests for other.print_point_report() which provides console output only

    def test_check_point_input_fail_1(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        message = ""
        try:
            # test only one exception raised by shared.check_feature_names() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            other._check_point_input(reference=reference, feature_names=[], target_names=None)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter feature_names must have one element per feature if not None.")

    def test_check_point_input_fail_2(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        message = ""
        try:
            # test only one exception raised by shared.check_feature_names() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            other._check_point_input(reference=reference, feature_names=None, target_names="target")
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter target_names must be a list of strings or None if reference belongs to a classifier."
        )

    def test_check_point_input_fail_3(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        message = ""
        try:
            # test only one exception raised by shared.check_feature_names() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            other._check_point_input(
                reference=reference,
                feature_names=None,
                target_names=[str(i) for i in range(1, reference["prediction"].shape[0])]
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "Parameter target_names must have one element per class if passing a list",
            "and reference belongs to a classifier."
        ]))

    def test_check_point_input_fail_4(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        reference["prediction"] = 3.0  # simulate regressor output
        message = ""
        try:
            # test only one exception raised by shared.check_feature_names() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            other._check_point_input(reference=reference, feature_names=None, target_names=["target 1", "target 2"])
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(
            message, "Parameter target_names must be a string or None if reference belongs to a regressor."
        )

    def test_check_point_input_1(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        ref_feature_names = check_feature_names(
            num_features=FEATURES.shape[1],
            feature_names=None,
            active_features=model.set_manager_.get_active_features()
        )
        feature_names, target_names, is_classifier_ = other._check_point_input(
            reference=reference,
            feature_names=None,
            target_names=None
        )
        self.assertEqual(feature_names, ref_feature_names)
        self.assertEqual(target_names, [str(i) for i in range(reference["prediction"].shape[0])])
        self.assertTrue(is_classifier_)

    def test_check_point_input_2(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        ref_feature_names = ["feature {}".format(i) for i in range(FEATURES.shape[1])]
        ref_target_names = ["class {}".format(i) for i in range(reference["prediction"].shape[0])]
        feature_names, target_names, is_classifier_ = other._check_point_input(
            reference=reference,
            feature_names=ref_feature_names,
            target_names=ref_target_names
        )
        self.assertEqual(feature_names, [ref_feature_names[i] for i in model.set_manager_.get_active_features()])
        self.assertEqual(target_names, ref_target_names)
        self.assertFalse(target_names is ref_target_names)  # ensure input list is copied
        self.assertTrue(is_classifier_)

    def test_check_point_input_3(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        reference["prediction"] = 3.0  # simulate regressor output
        ref_feature_names = ["feature {}".format(i) for i in range(FEATURES.shape[1])]
        feature_names, target_names, is_classifier_ = other._check_point_input(
            reference=reference,
            feature_names=ref_feature_names,
            target_names=None
        )
        self.assertEqual(feature_names, [ref_feature_names[i] for i in model.set_manager_.get_active_features()])
        self.assertEqual(target_names, "value")
        self.assertFalse(is_classifier_)

    def test_check_point_input_4(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        reference = other.choose_reference_point(features=FEATURES, model=model)
        reference["prediction"] = 3.0  # simulate regressor output
        ref_feature_names = ["feature {}".format(i) for i in range(FEATURES.shape[1])]
        feature_names, target_names, is_classifier_ = other._check_point_input(
            reference=reference,
            feature_names=ref_feature_names,
            target_names="target"
        )
        self.assertEqual(feature_names, [ref_feature_names[i] for i in model.set_manager_.get_active_features()])
        self.assertEqual(target_names, "target")
        self.assertFalse(is_classifier_)

    # no tests for write.write_report() which provides file output only

    def test_update_format_1(self):
        default = {"a": 1}
        result = write._update_format(format_=None, default=default)
        self.assertEqual(result, default)
        self.assertFalse(result is default)  # ensure default is copied

    def test_update_format_2(self):
        default = {"a": 1, "b": 2}
        result = write._update_format(format_={"a": 3}, default=default)
        self.assertEqual(result, {"a": 3, "b": 2})
