"""Unit tests for code in the benchmarks submodule.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from proset.benchmarks import reference


FEATURES = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
LABELS = np.array([0, 0, 1, 1, 2])
ETA = (1e-1, 1e-2)
ETA_ALT = 1e-3
NUM_ITER = (100, 1000)
NUM_ITER_ALT = 10
MAX_DEPTH = 10
COLSAMPLE_RANGE = (0.1, 0.9)
COLSAMPLE_RANGE_ALT = 0.75
SUBSAMPLE_RANGE = (0.2, 0.8)
SUBSAMPLE_RANGE_ALT = 1.0
STAGE_1_TRIALS = 100
NUM_FOLDS = 5
RANDOM_STATE = 12345
CV_RESULTS = {
    "mean_test_score": np.array([-0.8, -0.6, -0.58]),
    # sklearn cross-validation is always cast as a maximization problem, so test scores represent negative log-loss
    "std_test_score": np.array([0.05, 0.05, 0.05]),
    "param_max_depth": np.array([2, 3, 4])
}


# pylint: disable=too-few-public-methods
class MockEstimator:
    """Mock instance of an sklearn estimator.
    """

    @staticmethod
    def get_params():
        """Return parameters.

        :return: dict with estimator parameters
        """
        return {"max_depth": MAX_DEPTH, "colsample_bylevel": COLSAMPLE_RANGE_ALT, "subsample": SUBSAMPLE_RANGE_ALT}


# pylint: disable=too-few-public-methods
class MockSearch:
    """Mock instance of sklearn.model_selection.RandomizedSearchCV.
    """

    cv_results_ = CV_RESULTS
    estimator = MockEstimator()


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestBenchmarks(TestCase):
    """Unit tests for selected helper functions in proset.benchmarks.
    """

    # no tests for methods in auxiliary.py as they cover logging only

    # no tests for reference.fit_knn_classifier(), benchmark scripts serve as integration test

    # no tests for reference.fit_xgb_classifier(), benchmark scripts serve as integration test

    def test_check_xgb_classifier_settings_fail_1(self):
        message = ""
        try:
            # test only one exception raised by reference._check_ratios() to ensure it is called for eta; other
            # exceptions tested by the unit tests for that function
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=0.0,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must lie in (0.0, 1.0] if passing a single float.")

    def test_check_xgb_classifier_settings_fail_2(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=10.0,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must be integer if passing a single value.")

    def test_check_xgb_classifier_settings_fail_3(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=0,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must be positive if passing a single value.")

    def test_check_xgb_classifier_settings_fail_4(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=(10, 100, 1000),
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must have length 2 if passing a tuple.")

    def test_check_xgb_classifier_settings_fail_5(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=(10.0, 100),
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must contain integer values if passing a tuple.")

    def test_check_xgb_classifier_settings_fail_6(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=(10, 100.0),
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must contain integer values if passing a tuple.")

    def test_check_xgb_classifier_settings_fail_7(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=(0, 100),
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must contain positive values if passing a tuple.")

    def test_check_xgb_classifier_settings_fail_8(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=(10, 0),
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_iter must contain positive values if passing a tuple.")

    def test_check_xgb_classifier_settings_fail_9(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=10.0,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_depth must be integer.")

    def test_check_xgb_classifier_settings_fail_10(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=0,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_depth must be positive.")

    def test_check_xgb_classifier_settings_fail_11(self):
        message = ""
        try:
            # test only one exception raised by reference._check_ratios() to ensure it is called for colsample_range;
            # other exceptions tested by the unit tests for that function
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=0.0,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter colsample_range must lie in (0.0, 1.0] if passing a single float.")

    def test_check_xgb_classifier_settings_fail_12(self):
        message = ""
        try:
            # test only one exception raised by reference._check_ratios() to ensure it is called for colsample_range;
            # other exceptions tested by the unit tests for that function
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=0.0,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter subsample_range must lie in (0.0, 1.0] if passing a single float.")

    def test_check_xgb_classifier_settings_fail_13(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=100.0,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter stage_1_trials must be integer.")

    def test_check_xgb_classifier_settings_fail_14(self):
        message = ""
        try:
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=0,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter stage_1_trials must be positive.")

    def test_check_xgb_classifier_settings_1(self):
        result = reference._check_xgb_classifier_settings(
            labels=LABELS,
            eta=ETA,
            num_iter=NUM_ITER,
            max_depth=MAX_DEPTH,
            colsample_range=COLSAMPLE_RANGE,
            subsample_range=SUBSAMPLE_RANGE,
            stage_1_trials=STAGE_1_TRIALS,
            num_folds=NUM_FOLDS,
            random_state=RANDOM_STATE
        )
        self.assertEqual(len(result), 9)
        self.assertEqual(result["num_classes"], np.max(LABELS) + 1)
        self.assertEqual(result["eta"], ETA)
        self.assertEqual(result["num_iter"], NUM_ITER)
        self.assertEqual(result["max_depth"], MAX_DEPTH)
        self.assertEqual(result["colsample_range"], COLSAMPLE_RANGE)
        self.assertEqual(result["subsample_range"], SUBSAMPLE_RANGE)
        self.assertEqual(result["stage_1_trials"], STAGE_1_TRIALS)
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))
        self.assertEqual(result["splitter"].n_splits, NUM_FOLDS)
        self.assertTrue(isinstance(result["random_state"], np.random.RandomState))

    def test_check_xgb_classifier_settings_2(self):
        result = reference._check_xgb_classifier_settings(
            labels=LABELS,
            eta=ETA_ALT,
            num_iter=NUM_ITER_ALT,
            max_depth=MAX_DEPTH,
            colsample_range=COLSAMPLE_RANGE_ALT,
            subsample_range=SUBSAMPLE_RANGE_ALT,
            stage_1_trials=STAGE_1_TRIALS,
            num_folds=NUM_FOLDS,
            random_state=np.random.RandomState(RANDOM_STATE)
        )
        self.assertEqual(len(result), 9)
        self.assertEqual(result["num_classes"], np.max(LABELS) + 1)
        self.assertEqual(result["eta"], (ETA_ALT, ETA_ALT))
        self.assertEqual(result["num_iter"], (NUM_ITER_ALT, NUM_ITER_ALT))
        self.assertEqual(result["max_depth"], MAX_DEPTH)
        self.assertEqual(result["colsample_range"], COLSAMPLE_RANGE_ALT)
        self.assertEqual(result["subsample_range"], SUBSAMPLE_RANGE_ALT)
        self.assertEqual(result["stage_1_trials"], STAGE_1_TRIALS)
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))
        self.assertEqual(result["splitter"].n_splits, NUM_FOLDS)
        self.assertTrue(isinstance(result["random_state"], np.random.RandomState))

    def test_check_ratios_fail_1(self):
        message = ""
        try:
            reference._check_ratios(ratios=0.0, parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must lie in (0.0, 1.0] if passing a single float.")

    def test_check_ratios_fail_2(self):
        message = ""
        try:
            reference._check_ratios(ratios=1.1, parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must lie in (0.0, 1.0] if passing a single float.")

    def test_check_ratios_fail_3(self):
        message = ""
        try:
            reference._check_ratios(ratios=(0.5, 0.5, 0.5), parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must have length 2 if passing a tuple.")

    def test_check_ratios_fail_4(self):
        message = ""
        try:
            reference._check_ratios(ratios=(0.0, 0.5), parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must have both elements in (0.0, 1.0] if passing a tuple.")

    def test_check_ratios_fail_5(self):
        message = ""
        try:
            reference._check_ratios(ratios=(1.1, 0.5), parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must have both elements in (0.0, 1.0] if passing a tuple.")

    def test_check_ratios_fail_6(self):
        message = ""
        try:
            reference._check_ratios(ratios=(0.5, 0.0), parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must have both elements in (0.0, 1.0] if passing a tuple.")

    def test_check_ratios_fail_7(self):
        message = ""
        try:
            reference._check_ratios(ratios=(0.5, 1.1), parameter_name="eta", check_order=False)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must have both elements in (0.0, 1.0] if passing a tuple.")

    def test_check_ratios_fail_8(self):
        message = ""
        try:
            reference._check_ratios(ratios=(0.8, 0.5), parameter_name="eta", check_order=True)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter eta must have elements in strictly increasing order if passing a tuple.")

    @staticmethod
    def test_check_ratios_1():
        reference._check_ratios(ratios=0.5, parameter_name="eta", check_order=False)

    @staticmethod
    def test_check_ratios_2():
        reference._check_ratios(ratios=(0.5, 0.8), parameter_name="eta", check_order=True)

    # no tests for reference._fit_xgb_classifier_stage_1(), benchmark scripts serve as integration test

    def test_get_xgb_classifier_stage_1_parameters_1(self):
        fixed_para, search_para = reference._get_xgb_classifier_stage_1_parameters(
            reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        )
        self.assertEqual(len(fixed_para), 7)
        self.assertEqual(fixed_para["n_estimators"], NUM_ITER[0])
        self.assertEqual(fixed_para["use_label_encoder"], False)
        self.assertEqual(fixed_para["learning_rate"], ETA[0])
        self.assertTrue(np.issubdtype(type(fixed_para["random_state"]), np.integer))
        self.assertEqual(fixed_para["objective"], "multi:softprob")
        self.assertEqual(fixed_para["eval_metric"], "mlogloss")
        self.assertEqual(fixed_para["num_class"], np.max(LABELS) + 1)
        self.assertEqual(len(search_para), 3)
        self.assertEqual(search_para["max_depth"].a, 0)
        self.assertEqual(search_para["max_depth"].b, MAX_DEPTH - 1)
        self.assertEqual(search_para["colsample_bylevel"].kwds["loc"], COLSAMPLE_RANGE[0])
        self.assertEqual(search_para["colsample_bylevel"].kwds["scale"], COLSAMPLE_RANGE[1] - COLSAMPLE_RANGE[0])
        self.assertEqual(search_para["subsample"].kwds["loc"], SUBSAMPLE_RANGE[0])
        self.assertEqual(search_para["subsample"].kwds["scale"], SUBSAMPLE_RANGE[1] - SUBSAMPLE_RANGE[0])

    def test_get_xgb_classifier_stage_1_parameters_2(self):
        fixed_para, search_para = reference._get_xgb_classifier_stage_1_parameters(
            reference._check_xgb_classifier_settings(
                labels=np.array([0, 0, 1, 1]),  # binary classifier has different settings
                eta=ETA_ALT,
                num_iter=NUM_ITER_ALT,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE_ALT,
                subsample_range=SUBSAMPLE_RANGE_ALT,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            )
        )
        self.assertEqual(len(fixed_para), 8)
        self.assertEqual(fixed_para["n_estimators"], NUM_ITER_ALT)
        self.assertEqual(fixed_para["use_label_encoder"], False)
        self.assertEqual(fixed_para["learning_rate"], ETA_ALT)
        self.assertTrue(np.issubdtype(type(fixed_para["random_state"]), np.integer))
        self.assertEqual(fixed_para["objective"], "binary:logistic")
        self.assertEqual(fixed_para["eval_metric"], "logloss")
        self.assertEqual(fixed_para["colsample_bylevel"], COLSAMPLE_RANGE_ALT)
        self.assertEqual(fixed_para["subsample"], SUBSAMPLE_RANGE_ALT)
        self.assertEqual(len(search_para), 1)
        self.assertEqual(search_para["max_depth"].a, 0)
        self.assertEqual(search_para["max_depth"].b, MAX_DEPTH - 1)

    def test_get_objective_info_1(self):
        result = reference._get_objective_info(num_classes=2)
        self.assertEqual(result, {"objective": "binary:logistic", "eval_metric": "logloss"})

    def test_get_objective_info_2(self):
        result = reference._get_objective_info(num_classes=3)
        self.assertEqual(result, {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": 3})

    def test_get_xgb_classifier_stage_1_results(self):
        result = reference._get_xgb_classifier_stage_1_results(search=MockSearch(), search_para_names=["max_depth"])
        self.assertEqual(len(result), 7)
        np.testing.assert_array_equal(result["max_depth_grid"], CV_RESULTS["param_max_depth"])
        np.testing.assert_array_equal(
            result["colsample_grid"], COLSAMPLE_RANGE_ALT * np.ones(CV_RESULTS["param_max_depth"].shape[0])
        )
        np.testing.assert_array_equal(
            result["subsample_grid"], SUBSAMPLE_RANGE_ALT * np.ones(CV_RESULTS["param_max_depth"].shape[0])
        )
        np.testing.assert_array_equal(result["scores"], -CV_RESULTS["mean_test_score"])
        best_index = np.argmin(-CV_RESULTS["mean_test_score"])
        threshold = -CV_RESULTS["mean_test_score"][best_index] + CV_RESULTS["std_test_score"][best_index]
        self.assertEqual(result["threshold"], threshold)
        self.assertEqual(result["best_index"], best_index)
        self.assertEqual(result["selected_index"], np.nonzero(-CV_RESULTS["mean_test_score"] <= threshold)[0][0])

        # other outcomes of parameter search are tested by the unit tests for reference._update_candidates() below

    @staticmethod
    def test_update_candidates_1():
        ref_candidates = np.ones(3, dtype=bool)
        candidates, para_grid = reference._update_candidates(
            candidates=ref_candidates,
            search=MockSearch(),
            search_para_names={},
            para_name="max_depth",
            use_min=True
        )
        np.testing.assert_array_equal(candidates, ref_candidates)
        np.testing.assert_array_equal(para_grid, MAX_DEPTH * np.ones(3, dtype=int))

    @staticmethod
    def test_update_candidates_2():
        ref_candidates = np.ones(3, dtype=bool)
        selection = np.min(CV_RESULTS["param_max_depth"])
        candidates, para_grid = reference._update_candidates(
            candidates=ref_candidates,
            search=MockSearch(),
            search_para_names={"max_depth"},
            para_name="max_depth",
            use_min=True
        )
        np.testing.assert_array_equal(candidates, CV_RESULTS["param_max_depth"] == selection)
        np.testing.assert_array_equal(para_grid, CV_RESULTS["param_max_depth"])

    @staticmethod
    def test_update_candidates_3():
        ref_candidates = np.ones(3, dtype=bool)
        ref_grid = MockSearch.cv_results_["param_max_depth"]
        selection = np.max(ref_grid)
        candidates, para_grid = reference._update_candidates(
            candidates=ref_candidates,
            search=MockSearch,
            search_para_names={"max_depth"},
            para_name="max_depth",
            use_min=False
        )
        np.testing.assert_array_equal(candidates, ref_grid == selection)
        np.testing.assert_array_equal(para_grid, ref_grid)

    # no tests for reference._fit_xgb_classifier_stage_2(), benchmark scripts serve as integration test

    def test_get_xgb_classifier_stage_2_parameters_1(self):
        stage_1 = reference._get_xgb_classifier_stage_1_results(search=MockSearch(), search_para_names=["max_depth"])
        result = reference._get_xgb_classifier_stage_2_parameters(
            features=FEATURES,
            labels=LABELS,
            settings=reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=NUM_ITER,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            ),
            stage_1=stage_1
        )
        self.assertEqual(len(result), 6)
        self.assertEqual(len(result["params"]), 7)
        self.assertEqual(result["params"]["eta"], ETA[1])
        self.assertEqual(result["params"]["max_depth"], stage_1["max_depth_grid"][stage_1["selected_index"]])
        self.assertEqual(
            result["params"]["colsample_bylevel"], stage_1["colsample_grid"][stage_1["selected_index"]]
        )
        self.assertEqual(result["params"]["subsample"], stage_1["subsample_grid"][stage_1["selected_index"]])
        self.assertEqual(result["num_boost_round"], NUM_ITER[1])
        self.assertTrue(isinstance(result["folds"], StratifiedKFold))
        self.assertEqual(result["folds"].n_splits, NUM_FOLDS)
        self.assertTrue(result["stratified"])
        self.assertTrue(np.issubdtype(type(result["seed"]), np.integer))

    def test_get_xgb_classifier_stage_2_results_1(self):
        stage_1 = reference._get_xgb_classifier_stage_1_results(search=MockSearch(), search_para_names=["max_depth"])
        stage_2_para = reference._get_xgb_classifier_stage_2_parameters(
            features=FEATURES,
            labels=LABELS,
            settings=reference._check_xgb_classifier_settings(
                labels=LABELS,
                eta=ETA,
                num_iter=3,
                max_depth=MAX_DEPTH,
                colsample_range=COLSAMPLE_RANGE,
                subsample_range=SUBSAMPLE_RANGE,
                stage_1_trials=STAGE_1_TRIALS,
                num_folds=NUM_FOLDS,
                random_state=RANDOM_STATE
            ),
            stage_1=stage_1
        )
        search = pd.DataFrame({
            "test-mlogloss-mean": np.array([0.8, 0.6, 0.58]),
            "test-mlogloss-std": np.array([0.05, 0.05, 0.05])
        })
        result = reference._get_xgb_classifier_stage_2_results(search=search, stage_2_para=stage_2_para)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result["scores"], search["test-mlogloss-mean"].values)
        self.assertEqual(result["threshold"], 0.63)
        self.assertEqual(result["best_num_iter"], 3)
        self.assertEqual(result["selected_num_iter"], 2)

    # no tests for reference._fit_final_xgb_classifier(), benchmark scripts serve as integration test

    def test_get_xgb_classifier_final_parameters_1(self):
        result = reference._get_xgb_classifier_final_parameters(
            settings={"eta": (1e-1, 1e-2), "random_state": np.random.RandomState(), "num_classes": 2},
            stage_1={
                "max_depth_grid": np.array([2, 3, 4]),
                "subsample_grid": np.array([0.3, 0.2, 0.8]),
                "colsample_grid": np.array([0.5, 0.4, 0.1]),
                "selected_index": 1
            },
            stage_2={"selected_num_iter": 100}
        )
        self.assertEqual(len(result), 9)
        self.assertEqual(result["n_estimators"], 100)
        self.assertEqual(result["use_label_encoder"], False)
        self.assertEqual(result["max_depth"], 3)
        self.assertEqual(result["learning_rate"], 1e-2)
        self.assertEqual(result["subsample"], 0.2)
        self.assertEqual(result["colsample_bylevel"], 0.4)
        self.assertTrue(np.issubdtype(type(result["random_state"]), np.integer))
        self.assertEqual(result["objective"], "binary:logistic")
        self.assertEqual(result["eval_metric"], "logloss")

    # no tests for reference.print_xgb_classifier_report(), benchmark scripts serve as integration test

    # no tests for methods in samples.py, benchmark scripts serve as integration test
