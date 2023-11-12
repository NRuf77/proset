"""Unit tests for code in model.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from statsmodels.distributions.empirical_distribution import ECDF

from proset import ClassifierModel, shared
from proset.objectives.np_classifier_objective import NpClassifierObjective
from proset.objectives.tf_classifier_objective import TfClassifierObjective
from proset.set_manager import ClassifierSetManager
from test.test_np_objective import FEATURES, TARGET, COUNTS, WEIGHTS  # pylint: disable=wrong-import-order
from test.test_set_manager import BATCH_INFO  # pylint: disable=wrong-import-order


MARGINALS = COUNTS / np.sum(COUNTS)
LOG_PROBABILITY = np.log(MARGINALS[TARGET] + shared.LOG_OFFSET)
EXEMPT_CHECKS = [
    # messages from sklearn estimator checks that fail for proset so the exception is converted to a warning
    "For ClassifierModel, a zero sample_weight is not equivalent to removing the sample"
    # proset splits training data randomly, which means behavior depends on the order of samples even if the random seed
    # is fixed; the sklearn check does not provide samples in a consistent order for the two estimators being compared
]
EXEMPT_WARNING = "The sklearn estimator checks trigger an exception that proset cannot avoid:\n'{}'"


def _check_message_exempt(message):
    """Check whether an exception raised by sklearn estimator checks in the list of exceptions proset cannot avoid.

    :param message: string
    :return: boolean; whether the message is in the EXEMPT_CHECKS list
    """
    for reference in EXEMPT_CHECKS:
        if reference in message:
            return True
    return False  # pragma: no cover


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestClassifierModel(TestCase):
    """Unit tests for class ClassifierModel.

    The tests also cover abstract superclass Model.
    """

    @staticmethod
    def test_estimator_1():
        checks = check_estimator(ClassifierModel(), generate_only=True)
        for estimator, check in checks:
            try:
                check(estimator)
            except AssertionError as ex:
                message = ex.args[0]
                if _check_message_exempt(message):
                    warn(EXEMPT_WARNING.format(message))
                else:
                    raise ex  # pragma: no cover

    @staticmethod
    def test_estimator_2():
        checks = check_estimator(ClassifierModel(use_tensorflow=True), generate_only=True)
        for estimator, check in checks:
            try:
                check(estimator)
            except AssertionError as ex:
                message = ex.args[0]
                if _check_message_exempt(message):
                    warn(EXEMPT_WARNING.format(message))
                else:
                    raise ex  # pragma: no cover

    # no test for __init__() which only assigns public properties

    def test_fit_1(self):
        model = ClassifierModel(n_iter=1)
        model.fit(X=FEATURES, y=TARGET)
        self.assertEqual(model.set_manager_.num_batches, 1)
        model.fit(X=FEATURES, y=TARGET, warm_start=False)
        self.assertEqual(model.set_manager_.num_batches, 1)

    def test_fit_2(self):
        model = ClassifierModel(n_iter=1)
        model.fit(X=FEATURES.astype(np.float64), y=TARGET)  # check internal type conversion works
        self.assertEqual(model.set_manager_.num_batches, 1)
        model.fit(X=FEATURES.astype(np.float64), y=TARGET, warm_start=True)
        self.assertEqual(model.set_manager_.num_batches, 2)

    def test_fit_3(self):
        model = ClassifierModel(n_iter=1, use_tensorflow=True)
        model.fit(X=FEATURES, y=TARGET)
        self.assertEqual(model.set_manager_.num_batches, 1)

    # more extensive tests of fit() are performed by the sklearn test suite called in test_estimator_1() and
    # test_estimator_2()

    def test_check_hyperparameters_fail_1(self):
        message = ""
        try:
            ClassifierModel._check_hyperparameters(ClassifierModel(n_iter=1.0))
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter n_iter must be integer.")

    def test_check_hyperparameters_fail_2(self):
        message = ""
        try:
            ClassifierModel._check_hyperparameters(ClassifierModel(n_iter=-1))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter n_iter must not be negative.")

    def test_check_hyperparameters_fail_3(self):
        message = ""
        try:
            ClassifierModel._check_hyperparameters(ClassifierModel(solver_factr=0.0))
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must be positive.")

    @staticmethod
    def test_check_hyperparameters_1():
        ClassifierModel._check_hyperparameters(ClassifierModel(n_iter=0))

    def test_validate_arrays_fail_1(self):
        model = ClassifierModel()
        model.n_features_in_ = FEATURES.shape[1] + 1
        message = ""
        try:
            model._validate_arrays(X=FEATURES, y=TARGET, sample_weight=None, reset=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter X must have 5 columns.")

    def test_validate_arrays_fail_2(self):
        message = ""
        try:
            ClassifierModel()._validate_arrays(
                X=FEATURES,
                y=TARGET,
                sample_weight=np.ones(FEATURES.shape[0] + 1),
                reset=False
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter sample_weight must have one element per row of X if not None.")

    def test_validate_arrays_1(self):
        model = ClassifierModel()
        new_x, new_y, new_weight = model._validate_arrays(X=FEATURES, y=TARGET, sample_weight=None, reset=False)
        shared.check_float_array(x=new_x, name="new_x")
        np.testing.assert_array_equal(new_x, FEATURES)
        np.testing.assert_array_equal(new_y, TARGET)
        self.assertEqual(new_weight, None)
        self.assertTrue(hasattr(model, "label_encoder_"))
        np.testing.assert_array_equal(model.classes_, np.unique(TARGET))

    def test_validate_arrays_2(self):
        model = ClassifierModel()
        model.n_features_in_ = FEATURES.shape[1] + 1
        string_target = np.array([str(value) for value in TARGET])
        new_x, new_y, new_weight = model._validate_arrays(
            X=FEATURES.astype(np.float64),
            y=string_target,
            sample_weight=WEIGHTS,
            reset=True
        )
        shared.check_float_array(x=new_x, name="new_x")
        np.testing.assert_array_equal(new_x, FEATURES)
        np.testing.assert_array_equal(new_y, TARGET)  # converted to integer
        np.testing.assert_array_equal(new_weight, WEIGHTS)
        self.assertEqual(model.n_features_in_, FEATURES.shape[1])
        self.assertTrue(hasattr(model, "label_encoder_"))
        np.testing.assert_array_equal(model.classes_, np.unique(string_target))

    # function _validate_y() already tested by the above

    def test_get_compute_classes_1(self):
        # noinspection PyPep8Naming
        SetManager, Objective = ClassifierModel._get_compute_classes(False)
        self.assertTrue(SetManager is ClassifierSetManager)
        self.assertTrue(Objective is NpClassifierObjective)

    def test_get_compute_classes_2(self):
        # noinspection PyPep8Naming
        SetManager, Objective = ClassifierModel._get_compute_classes(True)
        self.assertTrue(SetManager is ClassifierSetManager)
        self.assertTrue(Objective is TfClassifierObjective)

    def test_parse_solver_status_1(self):
        result = ClassifierModel._parse_solver_status({"warnflag": 0})
        self.assertEqual(result, "converged")

    def test_parse_solver_status_2(self):
        result = ClassifierModel._parse_solver_status({"warnflag": 1})
        self.assertEqual(result, "reached limit on iterations or function calls")

    def test_parse_solver_status_3(self):
        result = ClassifierModel._parse_solver_status({"warnflag": 2, "task": "error"})
        self.assertEqual(result, "not converged (error)")

    def test_predict_fail_1(self):
        message = ""
        try:
            ClassifierModel().predict(X=FEATURES)
        except NotFittedError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "This ClassifierModel instance is not fitted yet.",
            "Call 'fit' with appropriate arguments before using this estimator."
        ]))

    @staticmethod
    def test_predict_1():
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        labels, familiarity = model.predict(X=FEATURES, compute_familiarity=True)
        np.testing.assert_array_equal(labels, np.argmax(COUNTS) * np.ones(FEATURES.shape[0], dtype=int))
        shared.check_float_array(x=familiarity, name="familiarity")
        np.testing.assert_array_equal(familiarity, np.zeros(FEATURES.shape[0], **shared.FLOAT_TYPE))

    def test_predict_2(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        labels = model.predict(X=FEATURES.astype(np.float64), n_iter=np.array([0]))
        # check internal type conversion works
        self.assertEqual(len(labels), 1)
        np.testing.assert_array_equal(labels[0], np.argmax(COUNTS) * np.ones(FEATURES.shape[0], dtype=int))

    def test_predict_3(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        labels, familiarity = model.predict(X=FEATURES, n_iter=np.array([0]), compute_familiarity=True)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(familiarity), 1)
        np.testing.assert_array_equal(labels[0], np.argmax(COUNTS) * np.ones(FEATURES.shape[0], dtype=int))
        shared.check_float_array(x=familiarity[0], name="familiarity[0]")
        np.testing.assert_array_equal(familiarity[0], np.zeros(FEATURES.shape[0], **shared.FLOAT_TYPE))

    @staticmethod
    def test_predict_4():
        model = ClassifierModel(n_iter=0, use_tensorflow=True)
        # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        labels, familiarity = model.predict(X=FEATURES, compute_familiarity=True)
        np.testing.assert_array_equal(labels, np.argmax(COUNTS) * np.ones(FEATURES.shape[0], dtype=int))
        shared.check_float_array(x=familiarity[0], name="familiarity[0]")
        np.testing.assert_array_equal(familiarity, np.zeros(FEATURES.shape[0], **shared.FLOAT_TYPE))

    # more extensive tests of predict() are performed by the sklearn test suite called in test_estimator_1() and
    # test_estimator_2()

    # function _compute_prediction() already covered by the above

    def test_score_fail_1(self):
        message = ""
        try:
            ClassifierModel().score(X=FEATURES, y=TARGET)
        except NotFittedError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "This ClassifierModel instance is not fitted yet.",
            "Call 'fit' with appropriate arguments before using this estimator."
        ]))

    def test_score_1(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        score = model.score(X=FEATURES, y=TARGET)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(score, np.mean(LOG_PROBABILITY), delta=1e-6)

    def test_score_2(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        score = model.score(X=FEATURES.astype(np.float64), y=TARGET, sample_weight=WEIGHTS, n_iter=np.array([0]))
        # check internal type conversion works
        # noinspection PyTypeChecker
        self.assertEqual(score.shape, (1, ))
        self.assertAlmostEqual(score[0], np.sum(LOG_PROBABILITY * WEIGHTS) / np.sum(WEIGHTS), delta=1e-6)

    def test_score_3(self):
        model = ClassifierModel(n_iter=0, use_tensorflow=True)
        # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        score = model.score(X=FEATURES, y=TARGET)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(score, np.mean(LOG_PROBABILITY), delta=1e-6)

    # more extensive tests of score() are performed by the sklearn test suite called in test_estimator_1() and
    # test_estimator_2()

    # function _compute_score() already covered by the above

    def test_predict_proba_fail_1(self):
        message = ""
        try:
            ClassifierModel().predict_proba(X=FEATURES)
        except NotFittedError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "This ClassifierModel instance is not fitted yet.",
            "Call 'fit' with appropriate arguments before using this estimator."
        ]))

    @staticmethod
    def test_predict_proba_1():
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        probabilities, familiarity = model.predict_proba(X=FEATURES, compute_familiarity=True)
        shared.check_float_array(x=probabilities, name="probabilities")
        np.testing.assert_allclose(probabilities, np.tile(MARGINALS, (FEATURES.shape[0], 1)), atol=1e-6)
        shared.check_float_array(x=familiarity, name="familiarity")
        np.testing.assert_allclose(familiarity, np.zeros(FEATURES.shape[0], **shared.FLOAT_TYPE), atol=1e-6)

    def test_predict_proba_2(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES.astype(np.float64), y=TARGET)  # check internal type conversion works
        probabilities = model.predict_proba(X=FEATURES, n_iter=np.array([0]))
        self.assertEqual(len(probabilities), 1)
        shared.check_float_array(x=probabilities[0], name="probabilities[0]")
        np.testing.assert_allclose(probabilities[0], np.tile(MARGINALS, (FEATURES.shape[0], 1)), atol=1e-6)

    def test_predict_proba_3(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        probabilities, familiarity = model.predict_proba(X=FEATURES, n_iter=np.array([0]), compute_familiarity=True)
        self.assertEqual(len(probabilities), 1)
        shared.check_float_array(x=probabilities[0], name="probabilities[0]")
        np.testing.assert_allclose(probabilities[0], np.tile(MARGINALS, (FEATURES.shape[0], 1)), atol=1e-6)
        self.assertEqual(len(familiarity), 1)
        shared.check_float_array(x=familiarity[0], name="familiarity[0]")
        np.testing.assert_allclose(familiarity[0], np.zeros(FEATURES.shape[0], **shared.FLOAT_TYPE), atol=1e-6)

    @staticmethod
    def test_predict_proba_4():
        model = ClassifierModel(n_iter=0, use_tensorflow=True)
        # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        probabilities, familiarity = model.predict_proba(X=FEATURES, compute_familiarity=True)
        shared.check_float_array(x=probabilities, name="probabilities")
        np.testing.assert_allclose(probabilities, np.tile(MARGINALS, (FEATURES.shape[0], 1)), atol=1e-6)
        shared.check_float_array(x=familiarity, name="familiarity")
        np.testing.assert_allclose(familiarity, np.zeros(FEATURES.shape[0], **shared.FLOAT_TYPE), atol=1e-6)

    # more extensive tests of predict_proba() are performed by the sklearn test suite called in test_estimator_1() and
    # test_estimator_2()

    def test_export_fail_1(self):
        message = ""
        model = ClassifierModel()
        try:
            model.export()
        except NotFittedError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "This ClassifierModel instance is not fitted yet.",
            "Call 'fit' with appropriate arguments before using this estimator."
        ]))

    def test_export_fail_2(self):
        message = ""
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        try:
            # test only one exception raised by _check_report_input() to ensure it is called; other exceptions tested by
            # the unit tests for that function
            model.export(train_names=[])
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter train_names must not be empty, pass None to use default sample names.")

    @staticmethod
    def test_export_1():
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        ref_baseline = model._make_baseline_for_export()
        batches = model.set_manager_.get_batches()
        ref_prototypes = model._make_prototype_report(batches=batches, train_names=None, compute_impact=False)
        feature_columns = model._check_report_input(
            n_iter=model.set_manager_.num_batches,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        ref_features = model._make_feature_report(
            batches=batches,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE),
            active_features=model.set_manager_.get_active_features(),
            include_similarities=False
        )
        ref_export = pd.concat([ref_prototypes, ref_features], axis=1)
        ref_export.sort_values(["batch", "prototype weight"], ascending=[True, False], inplace=True)
        ref_export = pd.concat([ref_baseline, ref_export], axis=0)
        ref_export.reset_index(drop=True, inplace=True)
        result = model.export()
        pd.testing.assert_frame_equal(result, ref_export)

    # behavior of different input parameters is tested for subordinate functions below

    def test_check_report_input_fail_1(self):
        message = ""
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        try:
            model._check_report_input(
                n_iter=1,
                train_names=[],
                feature_names=None,
                num_features=1,
                scale=None,
                offset=None,
                sample_name=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter train_names must not be empty, pass None to use default sample names.")

    def test_check_report_input_fail_2(self):
        message = ""
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        try:
            model._check_report_input(
                n_iter=1,
                train_names=[["sample_1"], ["sample_2"]],
                feature_names=None,
                num_features=1,
                scale=None,
                offset=None,
                sample_name=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "Parameter train_names must have as many elements as the number of batches to be evaluated",
            "if passing a list."
        ]))

    def test_check_report_input_fail_3(self):
        message = ""
        try:
            # test only one exception raised by shared.check_feature_names() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            ClassifierModel._check_report_input(
                n_iter=1,
                train_names=None,
                feature_names=None,
                num_features=0.0,
                scale=None,
                offset=None,
                sample_name=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_features must be integer.")

    def test_check_report_input_fail_4(self):
        message = ""
        try:
            # test only one exception raised by shared.check_scale_offset() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            ClassifierModel._check_report_input(
                n_iter=1,
                train_names=None,
                feature_names=None,
                num_features=1,
                scale=np.array([[1.0]]),
                offset=None,
                sample_name=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter scale must be a 1D array.")

    def test_check_report_input_1(self):
        feature_columns, include_original, scale, offset, sample_name = ClassifierModel._check_report_input(
            n_iter=1,
            train_names=None,
            feature_names=None,
            num_features=2,
            scale=None,
            offset=None,
            sample_name=None
        )
        self.assertEqual(feature_columns, [
            ["X0 weight", "X0 value", "X0 original", "X0 similarity"],
            ["X1 weight", "X1 value", "X1 original", "X1 similarity"]
        ])
        self.assertFalse(include_original)  # no custom scale or offset provided
        shared.check_float_array(x=scale, name="scale")
        np.testing.assert_array_equal(scale, np.ones(2, **shared.FLOAT_TYPE))
        shared.check_float_array(x=offset, name="offset")
        np.testing.assert_array_equal(offset, np.zeros(2, **shared.FLOAT_TYPE))
        self.assertEqual(sample_name, "new sample")

    def test_check_report_input_2(self):
        feature_columns, include_original, scale, offset, sample_name = ClassifierModel._check_report_input(
            n_iter=1,
            train_names=None,
            feature_names=["Y0", "Y1"],
            num_features=2,
            scale=np.array([0.5, 2.0]),
            offset=np.array([-1.0, 1.0]),
            sample_name="test sample"
        )
        self.assertEqual(feature_columns, [
            ["Y0 weight", "Y0 value", "Y0 original", "Y0 similarity"],
            ["Y1 weight", "Y1 value", "Y1 original", "Y1 similarity"]
        ])
        self.assertTrue(include_original)  # custom scale and offset provided
        shared.check_float_array(x=scale, name="scale")
        np.testing.assert_array_equal(scale, np.array([0.5, 2.0]))
        shared.check_float_array(x=offset, name="offset")
        np.testing.assert_array_equal(offset, np.array([-1.0, 1.0]))
        self.assertEqual(sample_name, "test sample")

    def test_make_prototype_report_1(self):
        model = ClassifierModel()
        result = model._make_prototype_report(batches=[], train_names=None, compute_impact=False)
        self.assertEqual(result.shape, (0, 5))
        self.assertEqual(list(result.columns), ["batch", "sample", "sample name", "target", "prototype weight"])

    def test_make_prototype_report_2(self):
        model = ClassifierModel()
        result = model._make_prototype_report(batches=[None], train_names=None, compute_impact=True)
        self.assertEqual(result.shape, (0, 7))
        self.assertEqual(
            list(result.columns),
            ["batch", "sample", "sample name", "target", "prototype weight", "similarity", "impact"]
        )

    def test_make_prototype_report_3(self):
        model = ClassifierModel()
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches()
        ref_batch = model._format_batch(batches[0], batch_index=0, train_names=None)
        result = model._make_prototype_report(batches=batches, train_names=None, compute_impact=False)
        self.assertEqual(result.shape, (2 * ref_batch.shape[0], 5))
        pd.testing.assert_frame_equal(result.iloc[:ref_batch.shape[0]], ref_batch)
        ref_batch["batch"] = 2
        result = result.iloc[ref_batch.shape[0]:].reset_index(drop=True)
        pd.testing.assert_frame_equal(result, ref_batch)

    @staticmethod
    def test_make_prototype_report_4():
        model = ClassifierModel()
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE)) + [None]
        train_names = ["training {}".format(j) for j in range(np.max(batches[0]["sample_index"]) + 1)]
        ref_batch = model._format_batch(batches[0], batch_index=0, train_names=train_names)
        result = model._make_prototype_report(batches=batches, train_names=train_names, compute_impact=True)
        pd.testing.assert_frame_equal(result, ref_batch)

    def test_make_prototype_report_5(self):
        model = ClassifierModel()
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches()
        num_samples = np.max(batches[0]["sample_index"]) + 1  # ensure there are enough names
        train_names = [
            ["batch_1_{}".format(i) for i in range(num_samples)],
            ["batch_2_{}".format(i) for i in range(num_samples)]
        ]
        ref_batch_1 = model._format_batch(batches[0], batch_index=0, train_names=train_names)
        ref_batch_2 = model._format_batch(batches[1], batch_index=1, train_names=train_names)
        result = model._make_prototype_report(batches=batches, train_names=train_names, compute_impact=False)
        self.assertEqual(result.shape, (ref_batch_1.shape[0] + ref_batch_2.shape[0], 5))
        pd.testing.assert_frame_equal(result.iloc[:ref_batch_1.shape[0]], ref_batch_1)
        result = result.iloc[ref_batch_1.shape[0]:].reset_index(drop=True)
        pd.testing.assert_frame_equal(result, ref_batch_2)

    def test_format_batch_1(self):
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batch = set_manager.get_batches()[0]
        result = ClassifierModel._format_batch(
            batch=batch,
            batch_index=0,
            train_names=None
        )
        num_prototypes = batch["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 5))
        np.testing.assert_array_equal(result["batch"].values, np.ones(num_prototypes, int))
        np.testing.assert_array_equal(result["sample"].values, batch["sample_index"])
        np.testing.assert_array_equal(
            result["sample name"].values, np.array(["sample {}".format(j) for j in batch["sample_index"]])
        )
        np.testing.assert_array_equal(result["target"].values, batch["target"])
        np.testing.assert_array_equal(result["prototype weight"].values, batch["prototype_weights"])

    def test_format_batch_2(self):
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batch = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE),)[0]
        train_names = ["training {}".format(j) for j in range(np.max(batch["sample_index"]) + 1)]
        result = ClassifierModel._format_batch(batch=batch, batch_index=0, train_names=train_names)
        num_prototypes = batch["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 7))
        np.testing.assert_array_equal(result["batch"].values, np.ones(num_prototypes, int))
        np.testing.assert_array_equal(result["sample"].values, batch["sample_index"])
        np.testing.assert_array_equal(
            result["sample name"].values, np.array([train_names[j] for j in batch["sample_index"]])
        )
        np.testing.assert_array_equal(result["target"].values, batch["target"])
        np.testing.assert_array_equal(result["prototype weight"].values, batch["prototype_weights"])
        similarity = np.prod(batch["similarities"], axis=1)
        np.testing.assert_almost_equal(result["similarity"], similarity)
        np.testing.assert_almost_equal(result["impact"], similarity * batch["prototype_weights"])

    def test_format_batch_3(self):
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batch = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE),)[0]
        train_names = [[], ["training {}".format(j) for j in range(np.max(batch["sample_index"]) + 1)]]
        result = ClassifierModel._format_batch(batch=batch, batch_index=1, train_names=train_names)
        num_prototypes = batch["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 7))
        np.testing.assert_array_equal(result["batch"].values, 2 * np.ones(num_prototypes, int))
        np.testing.assert_array_equal(result["sample"].values, batch["sample_index"])
        np.testing.assert_array_equal(
            result["sample name"].values, np.array([train_names[1][j] for j in batch["sample_index"]])
        )
        np.testing.assert_array_equal(result["target"].values, batch["target"])
        np.testing.assert_array_equal(result["prototype weight"].values, batch["prototype_weights"])
        similarity = np.prod(batch["similarities"], axis=1)
        np.testing.assert_almost_equal(result["similarity"], similarity)
        np.testing.assert_almost_equal(result["impact"], similarity * batch["prototype_weights"])

    @staticmethod
    def test_make_feature_report_1():
        model = ClassifierModel()
        num_features = 1
        scale = np.ones(num_features, **shared.FLOAT_TYPE)
        offset = np.zeros(num_features, **shared.FLOAT_TYPE)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=0,
            train_names=None,
            feature_names=None,
            num_features=num_features,
            scale=scale,
            offset=offset,
            sample_name=None
        )[0]
        result = model._make_feature_report(
            batches=[],
            feature_columns=feature_columns,
            include_original=False,
            scale=scale,
            offset=offset,
            active_features=np.zeros(0, dtype=int),  # no active features means nothing to report
            include_similarities=False
        )
        pd.testing.assert_frame_equal(result, pd.DataFrame())

    def test_make_feature_report_2(self):
        model = ClassifierModel()
        num_features = 3
        scale = np.ones(num_features, **shared.FLOAT_TYPE)
        offset = np.zeros(num_features, **shared.FLOAT_TYPE)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=1,
            train_names=None,
            feature_names=None,
            num_features=num_features,
            scale=scale,
            offset=offset,
            sample_name=None
        )[0]
        result = model._make_feature_report(
            batches=[None],  # no prototypes in batch means data frame has zero rows
            feature_columns=feature_columns,
            include_original=False,
            scale=scale,
            offset=offset,
            active_features=np.array([0, 2]),
            include_similarities=False
        )
        self.assertEqual(result.shape, (0, 4))  # two columns per active feature
        self.assertEqual(list(result.columns), ["X0 weight", "X0 value", "X2 weight", "X2 value"])

    def test_make_feature_report_3(self):
        model = ClassifierModel()
        num_features = 3
        scale = np.ones(num_features, **shared.FLOAT_TYPE)
        offset = np.zeros(num_features, **shared.FLOAT_TYPE)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=1,
            train_names=None,
            feature_names=None,
            num_features=num_features,
            scale=scale,
            offset=offset,
            sample_name=None
        )[0]
        result = model._make_feature_report(
            batches=[None],  # no prototypes in batch means data frame has zero rows
            feature_columns=feature_columns,
            include_original=True,
            scale=scale,
            offset=offset,
            active_features=np.array([0, 2]),
            include_similarities=True
        )
        self.assertEqual(result.shape, (0, 8))  # four columns per active feature
        self.assertEqual(list(result.columns), [
            "X0 weight", "X0 value", "X0 original", "X0 similarity",
            "X2 weight", "X2 value", "X2 original", "X2 similarity"
        ])

    @staticmethod
    def test_make_feature_report_4():
        model = ClassifierModel()
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches()
        num_features = np.max(batches[0]["active_features"]) + 1
        scale = np.ones(num_features, **shared.FLOAT_TYPE)
        offset = np.zeros(num_features, **shared.FLOAT_TYPE)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=set_manager.num_batches,
            train_names=None,
            feature_names=None,
            num_features=num_features,
            scale=scale,
            offset=offset,
            sample_name=None
        )[0]
        reference = pd.concat([
            ClassifierModel._format_feature(
                batches=batches,
                feature_index=index,
                feature_columns=feature_columns,
                include_original=False,
                scale=scale,
                offset=offset,
                include_similarities=False
            )
            for index in batches[0]["active_features"]
        ], axis=1)
        result = model._make_feature_report(
            batches=batches,
            feature_columns=feature_columns,
            include_original=False,
            scale=scale,
            offset=offset,
            active_features=batches[0]["active_features"],
            include_similarities=False
        )
        pd.testing.assert_frame_equal(result, reference)

    @staticmethod
    def test_make_feature_report_5():
        model = ClassifierModel()
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE))
        num_features = np.max(batches[0]["active_features"]) + 1
        scale = 2.0 * np.ones(num_features, **shared.FLOAT_TYPE)
        offset = -1.0 * np.ones(num_features, **shared.FLOAT_TYPE)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=set_manager.num_batches,
            train_names=None,
            feature_names=None,
            num_features=num_features,
            scale=scale,
            offset=offset,
            sample_name=None
        )[0]
        reference = pd.concat([
            ClassifierModel._format_feature(
                batches=batches,
                feature_index=index,
                feature_columns=feature_columns,
                include_original=True,
                scale=scale,
                offset=offset,
                include_similarities=True
            )
            for index in batches[0]["active_features"]
        ], axis=1)
        result = model._make_feature_report(
            batches=batches,
            feature_columns=feature_columns,
            include_original=True,
            scale=scale,
            offset=offset,
            active_features=batches[0]["active_features"],
            include_similarities=True
        )
        pd.testing.assert_frame_equal(result, reference)

    @staticmethod
    def test_make_feature_report_6():
        model = ClassifierModel()
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE)) + [None]
        num_features = np.max(batches[0]["active_features"]) + 1
        scale = 2.0 * np.ones(num_features, **shared.FLOAT_TYPE)
        offset = -1.0 * np.ones(num_features, **shared.FLOAT_TYPE)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=set_manager.num_batches,
            train_names=None,
            feature_names=None,
            num_features=num_features,
            scale=scale,
            offset=offset,
            sample_name=None
        )[0]
        reference = pd.concat([
            ClassifierModel._format_feature(
                batches=[batches[0]],  # second batch should have no contribution
                feature_index=index,
                feature_columns=feature_columns,
                include_original=True,
                scale=scale,
                offset=offset,
                include_similarities=True
            )
            for index in batches[0]["active_features"]
        ], axis=1)
        result = model._make_feature_report(
            batches=batches,
            feature_columns=feature_columns,
            include_original=True,
            scale=scale,
            offset=offset,
            active_features=batches[0]["active_features"],
            include_similarities=True
        )
        pd.testing.assert_frame_equal(result, reference)

    def test_format_feature_1(self):
        feature_columns = ClassifierModel._check_report_input(
            n_iter=0,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        result = ClassifierModel._format_feature(
            batches=[],
            feature_index=0,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE),
            include_similarities=False
        )
        self.assertEqual(result.shape, (0, 2))
        self.assertEqual(list(result.columns), feature_columns[0][:2])

    def test_format_feature_2(self):
        feature_columns = ClassifierModel._check_report_input(
            n_iter=1,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        result = ClassifierModel._format_feature(
            batches=[None],
            feature_index=0,
            feature_columns=feature_columns,
            include_original=True,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE),
            include_similarities=True
        )
        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(list(result.columns), feature_columns[0])

    def test_format_feature_3(self):
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches()
        feature_columns = ClassifierModel._check_report_input(
            n_iter=set_manager.num_batches,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        index = batches[0]["active_features"][0]
        result = ClassifierModel._format_feature(
            batches=batches,
            feature_index=index,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE),
            include_similarities=False
        )
        num_prototypes = batches[0]["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 2))
        np.testing.assert_array_equal(
            result["X{} weight".format(index)].values,
            batches[0]["feature_weights"][0] * np.ones(num_prototypes, **shared.FLOAT_TYPE)
        )
        np.testing.assert_array_equal(result["X{} value".format(index)].values, batches[0]["prototypes"][:, 0])

    def test_format_feature_4(self):
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE),)
        feature_columns = ClassifierModel._check_report_input(
            n_iter=set_manager.num_batches,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        index = batches[0]["active_features"][1]
        result = ClassifierModel._format_feature(
            batches=batches,
            feature_index=index,
            feature_columns=feature_columns,
            include_original=True,
            scale=2.0 * np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=-1.0 * np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            include_similarities=True
        )
        num_prototypes = batches[0]["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 4))
        np.testing.assert_array_equal(
            result["X{} weight".format(index)].values,
            batches[0]["feature_weights"][1] * np.ones(num_prototypes, **shared.FLOAT_TYPE)
        )
        np.testing.assert_array_equal(result["X{} value".format(index)].values, batches[0]["prototypes"][:, 1])
        np.testing.assert_array_equal(
            result["X{} original".format(index)].values, 2.0 * batches[0]["prototypes"][:, 1] - 1.0
        )
        np.testing.assert_array_equal(result["X{} similarity".format(index)].values, batches[0]["similarities"][:, 1])

    def test_format_feature_5(self):
        set_manager = ClassifierSetManager(target=TARGET, weights=None)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE))
        feature_columns = ClassifierModel._check_report_input(
            n_iter=set_manager.num_batches,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        index = 0
        for index in range(FEATURES.shape[0]):
            if index not in batches[0]["active_features"]:
                break
        if index in batches[0]["active_features"]:  # pragma: no cover
            raise RuntimeError("Constant BATCH_INFO from test_set_manager.py has no inactive features.")
        result = ClassifierModel._format_feature(
            batches=batches,
            feature_index=index,
            feature_columns=feature_columns,
            include_original=True,
            scale=2.0 * np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=-1.0 * np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            include_similarities=True
        )
        num_prototypes = batches[0]["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 4))
        self.assertTrue(np.all(pd.isna(result["X{} weight".format(index)].values)))
        self.assertTrue(np.all(pd.isna(result["X{} value".format(index)].values)))
        self.assertTrue(np.all(pd.isna(result["X{} original".format(index)].values)))
        self.assertTrue(np.all(pd.isna(result["X{} similarity".format(index)].values)))

    def test_make_baseline_for_export_1(self):
        model = ClassifierModel(n_iter=0)
        model.fit(X=FEATURES, y=TARGET)
        result = model._make_baseline_for_export()
        classes_int = np.unique(TARGET)
        classes_str = [str(label) for label in classes_int]
        self.assertEqual(result.shape, (len(classes_int), 5))
        self.assertTrue(np.all(pd.isna(result["batch"].values)))
        self.assertTrue(np.all(pd.isna(result["sample"].values)))
        self.assertEqual(list(result["sample name"]), model._format_class_labels(classes_str))
        np.testing.assert_array_equal(result["target"].values, classes_int)
        np.testing.assert_allclose(result["prototype weight"], MARGINALS, atol=1e-6)

    def test_format_class_labels_1(self):
        result = ClassifierModel._format_class_labels(["A", "B"])
        self.assertEqual(result, ["marginal probability class 'A'", "marginal probability class 'B'"])

    @staticmethod
    def test_explain_1():
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        active_features = model.set_manager_.get_active_features()
        feature_columns = model._check_report_input(
            n_iter=model.set_manager_.num_batches,
            train_names=None,
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        ref_baseline = model._make_baseline_for_explain(
            X=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE),
            y=TARGET[0],
            n_iter=1,
            familiarity=None,
            sample_name="new sample",
            include_features=True,
            active_features=active_features,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE)
        )
        batches = model.set_manager_.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE),)
        ref_prototypes = model._make_prototype_report(batches=batches, train_names=None, compute_impact=True)
        ref_contributions = model._make_contribution_report(ref_prototypes)
        ref_features = model._make_feature_report(
            batches=batches,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE),
            active_features=active_features,
            include_similarities=True
        )
        ref_explain = pd.concat([ref_prototypes, ref_contributions, ref_features], axis=1)
        ref_explain.sort_values(["batch", "impact"], ascending=[True, False], inplace=True)
        ref_explain = pd.concat([ref_baseline, ref_explain], axis=0)
        ref_explain.reset_index(drop=True, inplace=True)
        result = model.explain(X=FEATURES[0:1, :], y=TARGET[0])
        pd.testing.assert_frame_equal(result, ref_explain)

    # behavior of different input parameters is tested for subordinate functions above and below

    def test_make_contribution_report_1(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        prototype_report = model._make_prototype_report(
            batches=model.set_manager_.get_batches(features=FEATURES[0:1, :].astype(**shared.FLOAT_TYPE)),
            # argument features is necessary so batch info contains similarity data
            train_names=[str(i) for i in range(FEATURES.shape[0])],
            compute_impact=True
        )
        contributions, dominant_set = model._compute_contributions(
            impact=prototype_report["impact"].values,
            target=prototype_report["target"].values,
            marginals=model.set_manager_.marginals
        )
        result = model._make_contribution_report(prototype_report)
        self.assertEqual(result.shape, (model.set_manager_.get_num_prototypes(), 2 + np.max(TARGET)))
        np.testing.assert_array_equal(result["dominant set"].values, dominant_set)
        result.drop("dominant set", inplace=True, axis=1)
        np.testing.assert_array_equal(result.to_numpy(), contributions)

    @staticmethod
    def test_compute_contributions_1():
        contributions, dominant_set = ClassifierModel._compute_contributions(
            impact=np.zeros(0, dtype=float),  # no prototypes
            target=np.zeros(0, dtype=int),
            marginals=np.array([0.6, 0.4])
        )
        np.testing.assert_array_equal(contributions, np.zeros((0, 2), dtype=float))
        np.testing.assert_array_equal(dominant_set, np.zeros(0, dtype=int))

    @staticmethod
    def test_compute_contributions_2():
        contributions, dominant_set = ClassifierModel._compute_contributions(
            impact=np.array([1e-16]),  # contribution below numerical tolerance
            target=np.array([0]),
            marginals=np.array([0.6, 0.4])
        )
        np.testing.assert_array_equal(contributions, np.zeros((1, 2)))
        np.testing.assert_array_equal(dominant_set, np.zeros(1, dtype=int))

    @staticmethod
    def test_compute_contributions_3():
        impact = np.array([1e-5, 1e-5])
        # the contribution of the prototypes is negligible, the marginals determine the majority class
        target = np.array([0, 1])
        marginals = np.array([0.6, 0.4])
        scale = np.sum(impact) + 1.0
        scaled_impact = impact / scale
        ref_contributions = np.zeros((2, 2), dtype=float)
        for i in range(2):
            ref_contributions[i, target[i]] = scaled_impact[i]
        contributions, dominant_set = ClassifierModel._compute_contributions(
            impact=impact,
            target=target,
            marginals=marginals
        )
        np.testing.assert_allclose(contributions, ref_contributions, atol=1e-6)
        np.testing.assert_array_equal(dominant_set, np.array([0, 0]))

    @staticmethod
    def test_compute_contributions_4():
        impact = np.array([10.0, 10.0, 10.0, 1.0])
        # the 3rd prototype needs to be included in the dominant set as it has the same impact as the first two, the 4th
        # prototype has insufficient mass to affect the result
        target = np.array([0, 0, 1, 2])
        marginals = np.array([0.2, 0.3, 0.5])
        scale = np.sum(impact) + 1.0
        scaled_impact = impact / scale
        ref_contributions = np.zeros((4, 3), dtype=float)
        for i in range(4):
            ref_contributions[i, target[i]] = scaled_impact[i]
        contributions, dominant_set = ClassifierModel._compute_contributions(
            impact=impact,
            target=target,
            marginals=marginals
        )
        np.testing.assert_array_equal(contributions, ref_contributions)
        np.testing.assert_array_equal(dominant_set, np.array([1, 1, 1, 0]))

    def test_make_baseline_for_explain_1(self):
        model = ClassifierModel(n_iter=0)
        model.fit(X=FEATURES, y=TARGET)
        result = model._make_baseline_for_explain(
            X=FEATURES[0:1, :],
            y=TARGET[0],
            n_iter=0,
            familiarity=None,
            sample_name="new sample",
            include_features=False,
            active_features=np.zeros(0, dtype=int),
            feature_columns=[],
            include_original=False,
            scale=np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE),
            offset=np.zeros(FEATURES.shape[1], **shared.FLOAT_TYPE)
        )
        self.assertEqual(result.shape, (MARGINALS.shape[0] + 1, 11))
        self.assertTrue(np.all(pd.isna(result["batch"])))
        self.assertTrue(np.all(pd.isna(result["sample"])))
        names = ["new sample, prediction '{}'".format(np.argmax(MARGINALS))]
        names += ["marginal probability class '{}'".format(i) for i in range(MARGINALS.shape[0])]
        np.testing.assert_array_equal(result["sample name"].values, np.array(names))
        np.testing.assert_array_equal(result["target"].values, np.hstack([TARGET[0:1], np.arange(MARGINALS.shape[0])]))
        np.testing.assert_allclose(result["prototype weight"].values, np.hstack([np.NaN, MARGINALS]), atol=1e-6)
        np.testing.assert_allclose(
            result["similarity"].values,
            np.hstack([np.NaN, np.ones(MARGINALS.shape[0], **shared.FLOAT_TYPE)]),
            atol=1e-6
        )
        np.testing.assert_allclose(result["impact"].values, np.hstack([np.NaN, MARGINALS]), atol=1e-6)
        np.testing.assert_array_equal(
            result["dominant set"].values, np.hstack([np.NaN, np.ones(MARGINALS.shape[0], **shared.FLOAT_TYPE)])
        )
        for i, probability in enumerate(MARGINALS):
            reference = np.zeros(MARGINALS.shape[0], **shared.FLOAT_TYPE)
            reference[i] = probability
            np.testing.assert_allclose(
                result["p class {}".format(i)].values, np.hstack([probability, reference]), atol=1e-6
            )  # for zero batches, the predicted probabilities for the sample are equal to the marginals

    def test_make_baseline_for_explain_2(self):
        model = ClassifierModel()
        string_target = np.array(["T{}".format(i) for i in TARGET])
        model.fit(X=FEATURES, y=string_target)
        active_features = model.set_manager_.get_active_features()
        scale = 2.0 * np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE)
        offset = -1.0 * np.ones(FEATURES.shape[1], **shared.FLOAT_TYPE)
        sample_name = "new sample"
        feature_columns = model._check_report_input(
            n_iter=model.set_manager_.num_batches,
            train_names=None,
            feature_names=None,
            num_features=4,
            scale=scale,
            offset=offset,
            sample_name=sample_name
        )[0]
        prediction, familiarity = model.predict(X=FEATURES, compute_familiarity=True)
        probabilities = model.predict_proba(X=FEATURES[0:1, :])
        sample_familiarity = ECDF(familiarity)(familiarity[0])
        result = model._make_baseline_for_explain(
            X=FEATURES[0:1, :],
            y=string_target[0],
            n_iter=1,
            familiarity=familiarity,
            sample_name="new sample",
            include_features=True,
            active_features=active_features,
            feature_columns=feature_columns,
            include_original=True,
            scale=scale,
            offset=offset
        )
        self.assertEqual(result.shape, (MARGINALS.shape[0] + 1, 11 + 4 * active_features.shape[0]))
        self.assertTrue(np.all(pd.isna(result["batch"])))
        self.assertTrue(np.all(pd.isna(result["sample"])))
        names = ["{}, prediction '{}', familiarity {:.2f}".format(sample_name, prediction[0], sample_familiarity)]
        names += ["marginal probability class '{}'".format(label) for label in np.unique(string_target)]
        np.testing.assert_array_equal(result["sample name"].values, np.array(names))
        np.testing.assert_array_equal(result["target"].values, np.hstack([TARGET[0:1], np.arange(MARGINALS.shape[0])]))
        np.testing.assert_allclose(result["prototype weight"].values, np.hstack([np.NaN, MARGINALS]), atol=1e-6)
        np.testing.assert_allclose(
            result["similarity"].values,
            np.hstack([np.NaN, np.ones(MARGINALS.shape[0], **shared.FLOAT_TYPE)]),
            atol=1e-6
        )
        np.testing.assert_allclose(result["impact"].values, np.hstack([np.NaN, MARGINALS]), atol=1e-6)
        np.testing.assert_array_equal(
            result["dominant set"].values, np.hstack([np.NaN, np.ones(MARGINALS.shape[0], **shared.FLOAT_TYPE)])
        )
        for i, probability in enumerate(MARGINALS):
            reference = np.zeros(MARGINALS.shape[0], **shared.FLOAT_TYPE)
            reference[i] = probability / (familiarity[0] + 1.0)  # scale marginals to account for impact of prototypes
            np.testing.assert_allclose(
                result["p class {}".format(i)].values, np.hstack([probabilities[0, i], reference]), atol=1e-6
            )
        for i in active_features:
            self.assertTrue(np.all(pd.isna(result["X{} weight".format(i)].values)))
            np.testing.assert_allclose(
                result["X{} value".format(i)].values, np.array([FEATURES[0, i], np.NaN, np.NaN, np.NaN]), atol=1e-6
            )
            np.testing.assert_allclose(
                result["X{} original".format(i)].values,
                np.array([FEATURES[0, i] * 2.0 - 1.0, np.NaN, np.NaN, np.NaN]),
                atol=1e-6
            )
            self.assertTrue(np.all(pd.isna(result["X{} similarity".format(i)].values)))

    def test_shrink_fail_1(self):
        model = ClassifierModel()
        message = ""
        try:
            model.shrink()
        except NotFittedError as ex:
            message = ex.args[0]
        self.assertEqual(message, " ".join([
            "This ClassifierModel instance is not fitted yet.",
            "Call 'fit' with appropriate arguments before using this estimator."
        ]))

    def test_shrink_1(self):
        model = ClassifierModel(n_iter=0)
        model.fit(X=FEATURES, y=TARGET)
        model.shrink()
        self.assertTrue(hasattr(model, "active_features_"))
        np.testing.assert_array_equal(model.active_features_, np.zeros(0, dtype=int))
        self.assertEqual(model.n_features_in_, 0)

    def test_shrink_2(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        active_features = model.set_manager_.get_active_features()
        model.shrink()
        self.assertTrue(hasattr(model, "active_features_"))
        np.testing.assert_array_equal(model.active_features_, active_features)
        self.assertEqual(model.n_features_in_, active_features.shape[0])
