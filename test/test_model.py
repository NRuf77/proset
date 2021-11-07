"""Unit tests for code in model.py.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from statsmodels.distributions.empirical_distribution import ECDF

from proset import ClassifierModel
from proset.model import LOG_OFFSET
from proset.objective import ClassifierObjective
from proset.set_manager import ClassifierSetManager
from test.test_objective import FEATURES, TARGET, COUNTS, WEIGHTS  # pylint: disable=wrong-import-order
from test.test_set_manager import BATCH_INFO  # pylint: disable=wrong-import-order


MARGINALS = COUNTS / np.sum(COUNTS)
LOG_PROBABILITY = np.log(MARGINALS[TARGET] + LOG_OFFSET)


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestClassifierModel(TestCase):
    """Unit tests for class ClassifierModel.

    The tests also cover abstract superclass Model.
    """

    @staticmethod
    def test_estimator():
        check_estimator(ClassifierModel())

    # no test for __init__() which only assigns public properties

    def test_fit_1(self):
        model = ClassifierModel(n_iter=1)
        model.fit(X=FEATURES, y=TARGET)
        self.assertEqual(model.set_manager_.num_batches, 1)
        model.fit(X=FEATURES, y=TARGET, warm_start=False)
        self.assertEqual(model.set_manager_.num_batches, 1)

    def test_fit_2(self):
        model = ClassifierModel(n_iter=1)
        model.fit(X=FEATURES, y=TARGET)
        self.assertEqual(model.set_manager_.num_batches, 1)
        model.fit(X=FEATURES, y=TARGET, warm_start=True)
        self.assertEqual(model.set_manager_.num_batches, 2)

    # more extensive tests of predict() are performed by the sklearn test suite called in test_estimator()

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
            X=FEATURES,
            y=string_target,
            sample_weight=WEIGHTS,
            reset=True
        )
        np.testing.assert_array_equal(new_x, FEATURES)
        np.testing.assert_array_equal(new_y, TARGET)  # converted to integer
        np.testing.assert_array_equal(new_weight, WEIGHTS)
        self.assertEqual(model.n_features_in_, FEATURES.shape[1])
        self.assertTrue(hasattr(model, "label_encoder_"))
        np.testing.assert_array_equal(model.classes_, np.unique(string_target))

    # function _validate_y() already tested by the above

    def test_get_compute_classes_1(self):
        # noinspection PyPep8Naming
        SetManager, Objective = ClassifierModel._get_compute_classes()
        self.assertTrue(SetManager is ClassifierSetManager)
        self.assertTrue(Objective is ClassifierObjective)

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
        np.testing.assert_array_equal(familiarity, np.zeros(FEATURES.shape[0]))

    def test_predict_2(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        labels = model.predict(X=FEATURES, n_iter=np.array([0]))
        self.assertEqual(len(labels), 1)
        np.testing.assert_array_equal(labels[0], np.argmax(COUNTS) * np.ones(FEATURES.shape[0], dtype=int))

    def test_predict_3(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        labels, familiarity = model.predict(X=FEATURES, n_iter=np.array([0]), compute_familiarity=True)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(familiarity), 1)
        np.testing.assert_array_equal(labels[0], np.argmax(COUNTS) * np.ones(FEATURES.shape[0], dtype=int))
        np.testing.assert_array_equal(familiarity[0], np.zeros(FEATURES.shape[0]))

    # more extensive tests of predict() are performed by the sklearn test suite called in test_estimator()

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
        self.assertAlmostEqual(score, np.mean(LOG_PROBABILITY))

    def test_score_2(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        score = model.score(X=FEATURES, y=TARGET, sample_weight=WEIGHTS, n_iter=np.array([0]))
        # noinspection PyTypeChecker
        self.assertEqual(score.shape, (1, ))
        self.assertAlmostEqual(score[0], np.inner(LOG_PROBABILITY, WEIGHTS) / np.sum(WEIGHTS))

    # more extensive tests of score() are performed by the sklearn test suite called in test_estimator()

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
        np.testing.assert_array_equal(probabilities, np.tile(MARGINALS, (FEATURES.shape[0], 1)))
        np.testing.assert_array_equal(familiarity, np.zeros(FEATURES.shape[0]))

    def test_predict_proba_2(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        probabilities = model.predict_proba(X=FEATURES, n_iter=np.array([0]))
        self.assertEqual(len(probabilities), 1)
        np.testing.assert_array_equal(probabilities[0], np.tile(MARGINALS, (FEATURES.shape[0], 1)))

    def test_predict_proba_3(self):
        model = ClassifierModel(n_iter=0)  # constant model uses marginal distribution of target for predictions
        model.fit(X=FEATURES, y=TARGET)
        probabilities, familiarity = model.predict_proba(X=FEATURES, n_iter=np.array([0]), compute_familiarity=True)
        self.assertEqual(len(probabilities), 1)
        self.assertEqual(len(familiarity), 1)
        np.testing.assert_array_equal(probabilities[0], np.tile(MARGINALS, (FEATURES.shape[0], 1)))
        np.testing.assert_array_equal(familiarity[0], np.zeros(FEATURES.shape[0]))

    # more extensive tests of predict_proba() are performed by the sklearn test suite called in test_estimator()

    @staticmethod
    def test_export_1():
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        ref_baseline = model._make_baseline_for_export()
        batches = model.set_manager_.get_batches()
        ref_prototypes = model._make_prototype_report(batches=batches, train_names=None, compute_impact=False)
        feature_columns = model._check_report_input(
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
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1]),
            active_features=model.set_manager_.get_active_features(),
            include_similarities=False
        )
        ref_export = pd.concat([ref_prototypes, ref_features], axis=1)
        ref_export.sort_values(["batch", "prototype weight"], ascending=[True, False], inplace=True)
        ref_export = pd.concat([ref_baseline, ref_export], axis=0)
        ref_export.reset_index(drop=True, inplace=True)
        result = model.export()
        pd.testing.assert_frame_equal(result, ref_export)

    def test_check_report_input_fail_1(self):
        message = ""
        try:
            # test only one exception raised by shared.check_feature_names() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            ClassifierModel._check_report_input(
                feature_names=None,
                num_features=0.0,
                scale=None,
                offset=None,
                sample_name=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_features must be integer.")

    def test_check_report_input_fail_2(self):
        message = ""
        try:
            # test only one exception raised by shared.check_scale_offset() to ensure it is called; other exceptions
            # tested by the unit tests for that function
            ClassifierModel._check_report_input(
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
        np.testing.assert_array_equal(scale, np.ones(2))
        np.testing.assert_array_equal(offset, np.zeros(2))
        self.assertEqual(sample_name, "new sample")

    def test_check_report_input_2(self):
        feature_columns, include_original, scale, offset, sample_name = ClassifierModel._check_report_input(
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
        np.testing.assert_array_equal(scale, np.array([0.5, 2.0]))
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
        set_manager = ClassifierSetManager(target=TARGET)
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
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :]) + [None]
        train_names = ["training {}".format(j) for j in range(np.max(batches[0]["sample_index"]) + 1)]
        ref_batch = model._format_batch(batches[0], batch_index=0, train_names=train_names)
        result = model._make_prototype_report(batches=batches, train_names=train_names, compute_impact=True)
        pd.testing.assert_frame_equal(result, ref_batch)

    def test_format_batch_1(self):
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batch = set_manager.get_batches()[0]
        num_prototypes = batch["prototypes"].shape[0]
        result = ClassifierModel._format_batch(
            batch=batch,
            batch_index=0,
            train_names=None
        )
        self.assertEqual(result.shape, (num_prototypes, 5))
        np.testing.assert_array_equal(result["batch"].values, np.ones(num_prototypes))
        np.testing.assert_array_equal(result["sample"].values, batch["sample_index"])
        np.testing.assert_array_equal(
            result["sample name"].values, np.array(["sample {}".format(j) for j in batch["sample_index"]])
        )
        np.testing.assert_array_equal(result["target"].values, batch["target"])
        np.testing.assert_array_equal(result["prototype weight"].values, batch["prototype_weights"])

    def test_format_batch_2(self):
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batch = set_manager.get_batches(features=FEATURES[0:1, :])[0]
        num_prototypes = batch["prototypes"].shape[0]
        result = ClassifierModel._format_batch(
            batch=batch,
            batch_index=0,
            train_names=["training {}".format(j) for j in range(np.max(batch["sample_index"]) + 1)]
        )
        self.assertEqual(result.shape, (num_prototypes, 7))
        np.testing.assert_array_equal(result["batch"].values, np.ones(num_prototypes))
        np.testing.assert_array_equal(result["sample"].values, batch["sample_index"])
        np.testing.assert_array_equal(
            result["sample name"].values, np.array(["training {}".format(j) for j in batch["sample_index"]])
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
        scale = np.ones(num_features)
        offset = np.zeros(num_features)
        feature_columns = ClassifierModel._check_report_input(
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
        scale = np.ones(num_features)
        offset = np.zeros(num_features)
        feature_columns = ClassifierModel._check_report_input(
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
        scale = np.ones(num_features)
        offset = np.zeros(num_features)
        feature_columns = ClassifierModel._check_report_input(
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
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches()
        num_features = np.max(batches[0]["active_features"]) + 1
        scale = np.ones(num_features)
        offset = np.zeros(num_features)
        feature_columns = ClassifierModel._check_report_input(
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
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :])
        num_features = np.max(batches[0]["active_features"]) + 1
        scale = 2.0 * np.ones(num_features)
        offset = -1.0 * np.ones(num_features)
        feature_columns = ClassifierModel._check_report_input(
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
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :]) + [None]
        num_features = np.max(batches[0]["active_features"]) + 1
        scale = 2.0 * np.ones(num_features)
        offset = -1.0 * np.ones(num_features)
        feature_columns = ClassifierModel._check_report_input(
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
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1]),
            include_similarities=False
        )
        self.assertEqual(result.shape, (0, 2))
        self.assertEqual(list(result.columns), feature_columns[0][:2])

    def test_format_feature_2(self):
        feature_columns = ClassifierModel._check_report_input(
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
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1]),
            include_similarities=True
        )
        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(list(result.columns), feature_columns[0])

    def test_format_feature_3(self):
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches()
        feature_columns = ClassifierModel._check_report_input(
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
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1]),
            include_similarities=False
        )
        num_prototypes = batches[0]["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 2))
        np.testing.assert_array_equal(
            result["X{} weight".format(index)].values, batches[0]["feature_weights"][0] * np.ones(num_prototypes)
        )
        np.testing.assert_array_equal(result["X{} value".format(index)].values, batches[0]["prototypes"][:, 0])

    def test_format_feature_4(self):
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :])
        feature_columns = ClassifierModel._check_report_input(
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
            scale=2.0 * np.ones(FEATURES.shape[1]),
            offset=-1.0 * np.ones(FEATURES.shape[1]),
            include_similarities=True
        )
        num_prototypes = batches[0]["prototypes"].shape[0]
        self.assertEqual(result.shape, (num_prototypes, 4))
        np.testing.assert_array_equal(
            result["X{} weight".format(index)].values, batches[0]["feature_weights"][1] * np.ones(num_prototypes)
        )
        np.testing.assert_array_equal(result["X{} value".format(index)].values, batches[0]["prototypes"][:, 1])
        np.testing.assert_array_equal(
            result["X{} original".format(index)].values, 2.0 * batches[0]["prototypes"][:, 1] - 1.0
        )
        np.testing.assert_array_equal(result["X{} similarity".format(index)].values, batches[0]["similarities"][:, 1])

    def test_format_feature_5(self):
        set_manager = ClassifierSetManager(target=TARGET)
        set_manager.add_batch(BATCH_INFO)
        batches = set_manager.get_batches(features=FEATURES[0:1, :])
        feature_columns = ClassifierModel._check_report_input(
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
            scale=2.0 * np.ones(FEATURES.shape[1]),
            offset=-1.0 * np.ones(FEATURES.shape[1]),
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
        np.testing.assert_array_equal(result["prototype weight"], MARGINALS)

    def test_format_class_labels_1(self):
        result = ClassifierModel._format_class_labels(["A", "B"])
        self.assertEqual(result, ["marginal probability class 'A'", "marginal probability class 'B'"])

    @staticmethod
    def test_explain_1():
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        active_features = model.set_manager_.get_active_features()
        feature_columns = model._check_report_input(
            feature_names=None,
            num_features=FEATURES.shape[1],
            scale=None,
            offset=None,
            sample_name=None
        )[0]
        ref_baseline = model._make_baseline_for_explain(
            X=FEATURES[0:1, :],
            y=TARGET[0],
            n_iter=1,
            familiarity=None,
            sample_name="new sample",
            include_features=True,
            active_features=active_features,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1])
        )
        batches = model.set_manager_.get_batches(features=FEATURES[0:1, :])
        ref_prototypes = model._make_prototype_report(batches=batches, train_names=None, compute_impact=True)
        ref_contributions = model._make_contribution_report(ref_prototypes)
        ref_features = model._make_feature_report(
            batches=batches,
            feature_columns=feature_columns,
            include_original=False,
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1]),
            active_features=active_features,
            include_similarities=True
        )
        ref_explain = pd.concat([ref_prototypes, ref_contributions, ref_features], axis=1)
        ref_explain.sort_values(["batch", "impact"], ascending=[True, False], inplace=True)
        ref_explain = pd.concat([ref_baseline, ref_explain], axis=0)
        ref_explain.reset_index(drop=True, inplace=True)
        result = model.explain(X=FEATURES[0:1, :], y=TARGET[0])
        pd.testing.assert_frame_equal(result, ref_explain)

    def test_make_contribution_report_1(self):
        model = ClassifierModel()
        model.fit(X=FEATURES, y=TARGET)
        prototype_report = model._make_prototype_report(
            batches=model.set_manager_.get_batches(features=FEATURES[0:1, :]),
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
            impact=np.zeros(0),  # no prototypes
            target=np.zeros(0, dtype=int),
            marginals=np.array([0.6, 0.4])
        )
        np.testing.assert_array_equal(contributions, np.zeros((0, 2)))
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
        ref_contributions = np.zeros((2, 2))
        for i in range(2):
            ref_contributions[i, target[i]] = scaled_impact[i]
        contributions, dominant_set = ClassifierModel._compute_contributions(
            impact=impact,
            target=target,
            marginals=marginals
        )
        np.testing.assert_array_equal(contributions, ref_contributions)
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
        ref_contributions = np.zeros((4, 3))
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
            scale=np.ones(FEATURES.shape[1]),
            offset=np.zeros(FEATURES.shape[1])
        )
        self.assertEqual(result.shape, (MARGINALS.shape[0] + 1, 11))
        self.assertTrue(np.all(pd.isna(result["batch"])))
        self.assertTrue(np.all(pd.isna(result["sample"])))
        names = ["new sample, prediction '{}'".format(np.argmax(MARGINALS))]
        names += ["marginal probability class '{}'".format(i) for i in range(MARGINALS.shape[0])]
        np.testing.assert_array_equal(result["sample name"].values, np.array(names))
        np.testing.assert_array_equal(result["target"].values, np.hstack([TARGET[0:1], np.arange(MARGINALS.shape[0])]))
        np.testing.assert_array_equal(result["prototype weight"].values, np.hstack([np.NaN, MARGINALS]))
        np.testing.assert_array_equal(result["similarity"].values, np.hstack([np.NaN, np.ones(MARGINALS.shape[0])]))
        np.testing.assert_array_equal(result["impact"].values, np.hstack([np.NaN, MARGINALS]))
        np.testing.assert_array_equal(result["dominant set"].values, np.hstack([np.NaN, np.ones(MARGINALS.shape[0])]))
        for i, probability in enumerate(MARGINALS):
            reference = np.zeros(MARGINALS.shape[0])
            reference[i] = probability
            np.testing.assert_array_equal(result["p class {}".format(i)].values, np.hstack([probability, reference]))
            # for zero batches, the predicted probabilities for the sample are equal to the marginals

    def test_make_baseline_for_explain_2(self):
        model = ClassifierModel()
        string_target = np.array(["T{}".format(i) for i in TARGET])
        model.fit(X=FEATURES, y=string_target)
        active_features = model.set_manager_.get_active_features()
        scale = 2.0 * np.ones(FEATURES.shape[1])
        offset = -1.0 * np.ones(FEATURES.shape[1])
        sample_name = "new sample"
        feature_columns = model._check_report_input(
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
        np.testing.assert_array_equal(result["prototype weight"].values, np.hstack([np.NaN, MARGINALS]))
        np.testing.assert_array_equal(result["similarity"].values, np.hstack([np.NaN, np.ones(MARGINALS.shape[0])]))
        np.testing.assert_array_equal(result["impact"].values, np.hstack([np.NaN, MARGINALS]))
        np.testing.assert_array_equal(result["dominant set"].values, np.hstack([np.NaN, np.ones(MARGINALS.shape[0])]))
        for i, probability in enumerate(MARGINALS):
            reference = np.zeros(MARGINALS.shape[0])
            reference[i] = probability / (familiarity[0] + 1.0)  # scale marginals to account for impact of prototypes
            np.testing.assert_array_equal(
                result["p class {}".format(i)].values, np.hstack([probabilities[0, i], reference])
            )
        for i in active_features:
            self.assertTrue(np.all(pd.isna(result["X{} weight".format(i)].values)))
            np.testing.assert_array_equal(
                result["X{} value".format(i)].values, np.array([FEATURES[0, i], np.NaN, np.NaN, np.NaN])
            )
            np.testing.assert_array_equal(
                result["X{} original".format(i)].values, np.array([FEATURES[0, i] * 2.0 - 1.0, np.NaN, np.NaN, np.NaN])
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
