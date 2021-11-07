"""Unit tests for code in the utility submodule.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proset import ClassifierModel
from proset.shared import check_feature_names
import proset.utility.fit as fit
import proset.utility.other as other
import proset.utility.write as write
from test.test_objective import FEATURES, TARGET, WEIGHTS  # pylint: disable=wrong-import-order


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestUtility(TestCase):
    """Unit tests for selected helper functions in proset.utility.
    """

    # no tests for fit.select_hyperparameters(), benchmarks serve as integration test

    def test_process_select_settings_fail_1(self):
        message = ""
        try:
            fit._process_select_settings(
                model=ClassifierModel(),
                transform=None,
                lambda_v_range=-1.0,
                lambda_w_range=None,
                stage_1_trials=50,
                num_batch_grid=None,
                num_folds=5,
                num_jobs=None,
                random_state=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must not be negative if passing a single value.")
        # check only one exception message from _check_select_settings() to ensure it is called for lambda_v, full tests
        # for _check_select_settings() below

    def test_process_select_settings_1(self):
        model = ClassifierModel()
        result = fit._process_select_settings(
            model=model,
            transform=None,
            lambda_v_range=None,
            lambda_w_range=None,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=5,
            num_jobs=None,
            random_state=None
        )
        self.assertEqual(len(result), 10)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        self.assertEqual(result["prefix"], "")
        self.assertEqual(result["lambda_v_range"], fit.LAMBDA_V_RANGE)
        self.assertEqual(result["lambda_w_range"], fit.LAMBDA_W_RANGE)
        self.assertEqual(result["stage_1_trials"], 50)
        self.assertEqual(result["fit_mode"], fit.FitMode.BOTH)
        np.testing.assert_array_equal(result["num_batch_grid"], fit.NUM_BATCH_GRID)
        self.assertFalse(result["num_batch_grid"] is fit.NUM_BATCH_GRID)  # ensure mutable default is copied
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))  # classifier needs stratified splitter
        self.assertEqual(result["num_jobs"], None)
        self.assertTrue(isinstance(result["random_state"], np.random.RandomState))

    def test_process_select_settings_2(self):
        model = ClassifierModel()
        transform = StandardScaler()
        num_batch_grid = np.array([5])
        result = fit._process_select_settings(
            model=model,
            transform=transform,
            lambda_v_range=(1e-4, 1e-2),
            lambda_w_range=1e-7,
            stage_1_trials=50,
            num_batch_grid=num_batch_grid,
            num_folds=5,
            num_jobs=2,
            random_state=12345
        )
        self.assertEqual(len(result), 10)
        self.assertTrue(isinstance(result["model"], Pipeline))
        self.assertTrue(isinstance(result["model"]["model"], ClassifierModel))
        self.assertFalse(result["model"]["model"] is model)  # ensure original input is copied
        self.assertTrue(isinstance(result["model"]["transform"], StandardScaler))
        self.assertFalse(result["model"]["transform"] is transform)
        self.assertEqual(result["prefix"], "model__")
        self.assertEqual(result["lambda_v_range"], (1e-4, 1e-2))
        self.assertEqual(result["lambda_w_range"], 1e-7)
        self.assertEqual(result["stage_1_trials"], 50)
        self.assertEqual(result["fit_mode"], fit.FitMode.LAMBDA_V)
        np.testing.assert_array_equal(result["num_batch_grid"], num_batch_grid)
        self.assertFalse(result["num_batch_grid"] is num_batch_grid)
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))  # classifier needs stratified splitter
        self.assertEqual(result["num_jobs"], 2)
        self.assertTrue(isinstance(result["random_state"], np.random.RandomState))

    def test_check_select_settings_fail_1(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=-1.0,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must not be negative if passing a single value.")
        # check only one exception message from _check_lambda() to ensure it is called for lambda_v, full tests for
        # _check_lambda() below

    def test_check_select_settings_fail_2(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=-1.0,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_w must not be negative if passing a single value.")
        # check only one exception message from _check_lambda() to ensure it is called for lambda_w, full tests for
        # _check_lambda() below

    def test_check_select_settings_fail_3(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50.0,
                num_batch_grid=fit.NUM_BATCH_GRID,
                num_jobs=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter stage_1_trials must be integer.")

    def test_check_select_settings_fail_4(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=0,
                num_batch_grid=fit.NUM_BATCH_GRID,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter stage_1_trials must be positive.")

    def test_check_select_settings_fail_5(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=np.array([[5, 10]]),
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batch_grid must be a 1D array.")

    def test_check_select_settings_fail_6(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=np.array([5.0, 10.0]),
                num_jobs=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batch_grid must be an integer array.")

    def test_check_select_settings_fail_7(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=np.array([-5, 10]),
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batch_grid must not contain negative values.")

    def test_check_select_settings_fail_8(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=np.array([10, 5]),
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_batch_grid must contain strictly increasing values.")

    def test_check_select_settings_fail_9(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                num_jobs=2.0
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_jobs must be integer if not passing None.")

    def test_check_select_settings_fail_10(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                num_jobs=1
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_jobs must be greater than 1 if not passing None.")

    @staticmethod
    def test_check_select_settings_1():
        fit._check_select_settings(
            lambda_v_range=fit.LAMBDA_V_RANGE,
            lambda_w_range=fit.LAMBDA_W_RANGE,
            stage_1_trials=50,
            num_batch_grid=fit.NUM_BATCH_GRID,
            num_jobs=None
        )

    def test_check_lambda_fail_1(self):
        message = ""
        try:
            fit._check_lambda(lambda_range=-1.0, lambda_name="lambda_v")
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must not be negative if passing a single value.")

    def test_check_lambda_fail_2(self):
        message = ""
        try:
            fit._check_lambda(lambda_range=(0.0, 1e-3, 1e-1), lambda_name="lambda_v")
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must have length two if passing a tuple.")

    def test_check_lambda_fail_3(self):
        message = ""
        try:
            fit._check_lambda(lambda_range=(-1.0, 1e-1), lambda_name="lambda_v")
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must not contain negative values if passing a tuple.")

    def test_check_lambda_fail_4(self):
        message = ""
        try:
            fit._check_lambda(lambda_range=(1e-1, 1e-3), lambda_name="lambda_v")
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v must contain strictly increasing values if passing a tuple.")

    @staticmethod
    def test_check_lambda_1():
        fit._check_lambda(lambda_range=1e-3, lambda_name="lambda_v")

    @staticmethod
    def test_check_lambda_2():
        fit._check_lambda(lambda_range=(1e-3, 1e-1), lambda_name="lambda_v")

    def test_get_fit_mode_1(self):
        result = fit._get_fit_mode(lambda_v_range=1e-3, lambda_w_range=1e-8)
        self.assertEqual(result, fit.FitMode.NEITHER)

    def test_get_fit_mode_2(self):
        result = fit._get_fit_mode(lambda_v_range=fit.LAMBDA_V_RANGE, lambda_w_range=1e-8)
        self.assertEqual(result, fit.FitMode.LAMBDA_V)

    def test_get_fit_mode_3(self):
        result = fit._get_fit_mode(lambda_v_range=1e-3, lambda_w_range=fit.LAMBDA_W_RANGE)
        self.assertEqual(result, fit.FitMode.LAMBDA_W)

    def test_get_fit_mode_4(self):
        result = fit._get_fit_mode(lambda_v_range=fit.LAMBDA_V_RANGE, lambda_w_range=fit.LAMBDA_W_RANGE)
        self.assertEqual(result, fit.FitMode.BOTH)

    # no tests for fit._execute_stage_1(), benchmarks serve as integration test

    def test_make_stage_1_plan_1(self):
        model = ClassifierModel()
        settings = fit._process_select_settings(
            model=model,
            transform=None,
            lambda_v_range=None,
            lambda_w_range=None,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=3,
            num_jobs=None,
            random_state=None
        )
        plan, lambdas = fit._make_stage_1_plan(settings=settings, features=FEATURES, target=TARGET, weights=WEIGHTS)
        self.assertEqual(len(plan), 3 * 50)
        self.assertEqual(len(plan[0]), 9)
        self.assertTrue(isinstance(plan[0]["model"], ClassifierModel))
        self.assertFalse(plan[0]["model"] is model)  # ensure original input is copied
        np.testing.assert_array_equal(plan[0]["features"], FEATURES)
        np.testing.assert_array_equal(plan[0]["target"], TARGET)
        np.testing.assert_array_equal(plan[0]["weights"], WEIGHTS)
        self.assertEqual(plan[0]["prefix"], "")
        self.assertEqual(plan[0]["fold"], 0)
        self.assertEqual(plan[0]["train_ix"].shape, TARGET.shape)
        self.assertEqual(plan[0]["trial"], 0)
        self.assertEqual(len(plan[0]["parameters"]), 3)
        self.assertTrue(fit.LAMBDA_V_RANGE[0] <= plan[0]["parameters"]["lambda_v"] <= fit.LAMBDA_V_RANGE[1])
        self.assertTrue(fit.LAMBDA_W_RANGE[0] <= plan[0]["parameters"]["lambda_w"] <= fit.LAMBDA_W_RANGE[1])
        self.assertTrue(isinstance(plan[0]["parameters"]["random_state"], np.random.RandomState))
        reference = [{
            "lambda_v": plan[i]["parameters"]["lambda_v"],
            "lambda_w": plan[i]["parameters"]["lambda_w"]
        } for i in range(50)]
        self.assertEqual(lambdas, reference)

    def test_sample_lambda_1(self):
        result = fit._sample_lambda(
            lambda_range=1e-3,
            trials=10,
            do_randomize=False,
            random_state=np.random.RandomState()
        )
        self.assertEqual(result.shape, (10, ))
        self.assertTrue(np.all(result == 1e-3))

    def test_sample_lambda_2(self):
        result = fit._sample_lambda(
            lambda_range=fit.LAMBDA_V_RANGE,
            trials=10,
            do_randomize=False,
            random_state=np.random.RandomState()
        )
        self.assertEqual(result.shape, (10, ))
        np.testing.assert_almost_equal(
            result, np.logspace(np.log10(fit.LAMBDA_V_RANGE[0]), np.log10(fit.LAMBDA_V_RANGE[1]), 10)
        )

    def test_sample_lambda_3(self):
        result = fit._sample_lambda(
            lambda_range=fit.LAMBDA_V_RANGE,
            trials=10,
            do_randomize=True,
            random_state=np.random.RandomState()
        )
        self.assertEqual(result.shape, (10, ))
        self.assertTrue(np.all(np.logical_and(result >= fit.LAMBDA_V_RANGE[0], result <= fit.LAMBDA_V_RANGE[1])))

    def test_sample_random_state_1(self):
        random_state = np.random.RandomState(12345)
        result = fit._sample_random_state(size=1, random_state=random_state)
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result[0], np.random.RandomState))
        self.assertFalse(result[0] is random_state)

    def test_make_train_ix(self):
        result = fit._make_train_ix(features=FEATURES, target=TARGET, splitter=StratifiedKFold(n_splits=3))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, TARGET.shape)
        self.assertEqual(result[1].shape, TARGET.shape)
        self.assertEqual(result[2].shape, TARGET.shape)
        np.testing.assert_array_equal(
            result[0].astype(int) + result[1].astype(int) + result[2].astype(int),
            2 * np.ones(TARGET.shape[0], dtype=int)
        )  # every observation is included in two sets of training indices

    # no tests for fit._execute_plan(), benchmarks serve as integration test

    # no tests for fit._fit_stage_1(), benchmarks serve as integration test

    # no tests for fit._fit_model(), benchmarks serve as integration test

    @staticmethod
    def test_collect_stage_1_1():
        scores = np.array([[1, 2, 3], [4, 5, 6]])
        score_list = [(i, j, scores[i, j]) for i in range(scores.shape[0]) for j in range(scores.shape[1])]
        result = fit._collect_stage_1(results=score_list, num_folds=scores.shape[0], trials=scores.shape[1])
        np.testing.assert_array_equal(result, scores)

    def test_evaluate_stage_1_1(self):
        scores = np.array([[0.0, 0.9, 0.95], [0.1, 1.0, 0.9]])
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0, ddof=1)
        result = fit._evaluate_stage_1(
            scores=scores,
            parameters=[
                {"lambda_v": 1e-4, "lambda_w": 1e-8},
                {"lambda_v": 1e-3, "lambda_w": 1e-8},
                {"lambda_v": 1e-2, "lambda_w": 1e-8}
            ],
            fit_mode=fit.FitMode.LAMBDA_V
        )
        self.assertEqual(len(result), 6)
        np.testing.assert_array_equal(result["lambda_grid"], np.array([[1e-4, 1e-8], [1e-3, 1e-8], [1e-2, 1e-8]]))
        self.assertEqual(result["fit_mode"], fit.FitMode.LAMBDA_V)
        np.testing.assert_almost_equal(result["scores"], mean)
        self.assertEqual(result["threshold"], mean[1] - std[1])
        self.assertEqual(result["best_index"], 1)
        self.assertEqual(result["selected_index"], 2)

    def test_compute_stats_1(self):
        scores = np.array([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0]])
        result = fit._compute_stats(scores)
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result["mean"], np.array([0.5, 1.0, 1.5]))
        np.testing.assert_array_equal(result["best_index"], 2)
        self.assertAlmostEqual(result["threshold"], 1.5 - np.sqrt(0.5))

    def test_fake_stage_1_1(self):
        result = fit._fake_stage_1(lambda_v=1e-3, lambda_w=1e-8)
        self.assertEqual(len(result), 6)
        np.testing.assert_array_equal(result["lambda_grid"], np.array([[1e-3, 1e-8]]))
        self.assertEqual(result["fit_mode"], fit.FitMode.NEITHER)
        np.testing.assert_array_equal(result["scores"], np.array(np.NaN))
        self.assertTrue(pd.isna(result["threshold"]))
        self.assertEqual(result["best_index"], 0)
        self.assertEqual(result["selected_index"], 0)

    # no tests for fit._execute_stage_2(), benchmarks serve as integration test

    def test_make_stage_2_plan_1(self):
        model = ClassifierModel()
        settings = fit._process_select_settings(
            model=model,
            transform=None,
            lambda_v_range=None,
            lambda_w_range=None,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=3,
            num_jobs=None,
            random_state=None
        )
        result = fit._make_stage_2_plan(settings=settings, features=FEATURES, target=TARGET, weights=WEIGHTS)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 8)
        self.assertTrue(isinstance(result[0]["model"], ClassifierModel))
        self.assertFalse(result[0]["model"] is model)  # ensure original input is copied
        np.testing.assert_array_equal(result[0]["features"], FEATURES)
        np.testing.assert_array_equal(result[0]["target"], TARGET)
        np.testing.assert_array_equal(result[0]["weights"], WEIGHTS)
        np.testing.assert_array_equal(result[0]["num_batch_grid"], fit.NUM_BATCH_GRID)
        self.assertEqual(result[0]["prefix"], "")
        self.assertEqual(result[0]["train_ix"].shape, TARGET.shape)
        self.assertEqual(len(result[0]["parameters"]), 2)
        self.assertEqual(result[0]["parameters"]["n_iter"], fit.NUM_BATCH_GRID[-1])
        self.assertTrue(isinstance(result[0]["parameters"]["random_state"], np.random.RandomState))

    # no tests for fit._fit_stage_2(), benchmarks serve as integration test

    def test_evaluate_stage_2_1(self):
        scores = np.array([[0.0, 0.95, 0.9], [0.1, 0.9, 1.0]])
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0, ddof=1)
        result = fit._evaluate_stage_2(
            scores=scores,
            num_batch_grid=fit.NUM_BATCH_GRID
        )
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(result["num_batch_grid"], fit.NUM_BATCH_GRID)
        np.testing.assert_almost_equal(result["scores"], mean)
        self.assertEqual(result["threshold"], mean[2] - std[2])
        self.assertEqual(result["best_index"], 2)
        self.assertEqual(result["selected_index"], 1)

    def test_fake_stage_2_1(self):
        num_batch_grid = np.array([1])
        result = fit._fake_stage_2(num_batch_grid)
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(result["num_batch_grid"], num_batch_grid)
        np.testing.assert_array_equal(result["scores"], np.array(np.NaN))
        self.assertTrue(pd.isna(result["threshold"]))
        self.assertEqual(result["best_index"], 0)
        self.assertEqual(result["selected_index"], 0)

    # no tests for other.print_hyperparameter_report() which provides console output only

    # no tests for other.print_feature_report() which provides console output only

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
