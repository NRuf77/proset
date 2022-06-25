"""Unit tests for code in the utility submodule.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from proset import ClassifierModel
from proset.shared import check_feature_names
import proset.utility.fit as fit
import proset.utility.other as other
import proset.utility.write as write
from test.test_np_objective import FEATURES, TARGET, WEIGHTS  # pylint: disable=wrong-import-order


EXTENDED_FEATURES = np.vstack([FEATURES, FEATURES, FEATURES])
# sample data is too small for some tests involving cross-validation
EXTENDED_TARGET = np.hstack([TARGET, TARGET, TARGET])
EXTENDED_WEIGHTS = np.hstack([WEIGHTS, WEIGHTS, WEIGHTS])
EXTENDED_CV_GROUPS = np.hstack([np.zeros_like(TARGET), np.ones_like(TARGET), 2 * np.ones_like(TARGET)])


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestUtility(TestCase):
    """Unit tests for selected helper functions in proset.utility.
    """

    def test_select_hyperparameters_1(self):
        model = ClassifierModel()
        result = fit.select_hyperparameters(
            model=model,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        check_is_fitted(result["model"])
        self.assertEqual(sorted(list(result["stage_1"].keys())), [
            "best_index", "fit_mode", "lambda_grid", "scores", "selected_index", "threshold"
        ])
        self.assertEqual(sorted(list(result["stage_2"].keys())), [
            "best_index", "num_batch_grid", "scores", "selected_index", "threshold"
        ])
        # benchmark cases serve as integration test where results can be reviews in detail

    def test_select_hyperparameters_2(self):
        model = ClassifierModel()
        transform = StandardScaler()
        result = fit.select_hyperparameters(
            model=model,
            features=EXTENDED_FEATURES,
            target=list(EXTENDED_TARGET),
            # in accordance with sklearn convention, target is not required to be a numpy array
            weights=EXTENDED_WEIGHTS,
            cv_groups=EXTENDED_CV_GROUPS,
            transform=transform,
            lambda_v_range=(1e-4, 1e-2),
            lambda_w_range=(1e-9, 1e-7),
            stage_1_trials=10,
            num_batch_grid=np.array([0, 5, 10]),
            num_folds=3,
            solver_factr=(1e10, 1e8),
            chunks=2,
            num_jobs=None,
            random_state=12345
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(isinstance(result["model"], Pipeline))
        check_is_fitted(result["model"])
        self.assertTrue(isinstance(result["model"]["transform"], StandardScaler))
        self.assertFalse(result["model"]["transform"] is transform)  # ensure original input is copied
        np.testing.assert_allclose(result["model"]["transform"].mean_, np.mean(EXTENDED_FEATURES, axis=0), atol=1e-6)
        np.testing.assert_allclose(result["model"]["transform"].var_, np.var(EXTENDED_FEATURES, axis=0), atol=1e-6)
        self.assertTrue(isinstance(result["model"]["model"], ClassifierModel))
        self.assertFalse(result["model"]["model"] is model)
        self.assertEqual(sorted(list(result["stage_1"].keys())), [
            "best_index", "fit_mode", "lambda_grid", "scores", "selected_index", "threshold"
        ])
        self.assertEqual(sorted(list(result["stage_2"].keys())), [
            "best_index", "num_batch_grid", "scores", "selected_index", "threshold"
        ])
        self.assertEqual(len(result["chunk_ix"]), result["model"]["model"].set_manager_.num_batches)

    def test_select_hyperparameters_3(self):
        model = ClassifierModel()
        transform = StandardScaler()
        result = fit.select_hyperparameters(
            model=model,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            transform=transform,
            lambda_v_range=1e-3,
            lambda_w_range=1e-8,
            num_batch_grid=np.array([5])
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(isinstance(result["model"], Pipeline))
        check_is_fitted(result["model"])
        self.assertTrue(isinstance(result["model"]["transform"], StandardScaler))
        self.assertFalse(result["model"]["transform"] is transform)  # ensure original input is copied
        np.testing.assert_allclose(result["model"]["transform"].mean_, np.mean(EXTENDED_FEATURES, axis=0), atol=1e-6)
        np.testing.assert_allclose(result["model"]["transform"].var_, np.var(EXTENDED_FEATURES, axis=0), atol=1e-6)
        self.assertTrue(isinstance(result["model"]["model"], ClassifierModel))
        self.assertFalse(result["model"]["model"] is model)
        self.assertEqual(sorted(list(result["stage_1"].keys())), [
            "best_index", "fit_mode", "lambda_grid", "scores", "selected_index", "threshold"
        ])
        self.assertEqual(sorted(list(result["stage_2"].keys())), [
            "best_index", "num_batch_grid", "scores", "selected_index", "threshold"
        ])

    def test_select_hyperparameters_4(self):
        model = ClassifierModel()
        result = fit.select_hyperparameters(
            model=model,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            stage_1_trials=10,
            num_jobs=2
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        check_is_fitted(result["model"])
        self.assertEqual(sorted(list(result["stage_1"].keys())), [
            "best_index", "fit_mode", "lambda_grid", "scores", "selected_index", "threshold"
        ])
        self.assertEqual(sorted(list(result["stage_2"].keys())), [
            "best_index", "num_batch_grid", "scores", "selected_index", "threshold"
        ])

    def test_process_cv_groups_fail_1(self):
        message = ""
        try:
            fit._process_cv_groups(cv_groups=np.ones((TARGET.shape[0], 1), dtype=int), target=TARGET, classify=True)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter cv_groups must be a 1D array.")

    def test_process_cv_groups_fail_2(self):
        message = ""
        try:
            fit._process_cv_groups(cv_groups=np.ones(TARGET.shape[0] - 1, dtype=int), target=TARGET, classify=True)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter cv_groups must have one element for each sample.")

    def test_process_cv_groups_fail_3(self):
        message = ""
        try:
            fit._process_cv_groups(cv_groups=np.ones(TARGET.shape[0], dtype=float), target=TARGET, classify=True)
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter cv_groups must be integer.")

    def test_process_cv_groups_1(self):
        result = fit._process_cv_groups(cv_groups=None, target=TARGET, classify=True)
        self.assertEqual(result, None)

    @staticmethod
    def test_process_cv_groups_2():
        cv_groups = np.ones_like(TARGET, dtype=int)
        result = fit._process_cv_groups(cv_groups=cv_groups, target=TARGET, classify=True)
        reference = pd.DataFrame({"index": np.arange(TARGET.shape[0]), "cv_group": cv_groups, "target": TARGET})
        pd.testing.assert_frame_equal(result, reference)

    @staticmethod
    def test_process_cv_groups_3():
        cv_groups = np.ones_like(TARGET, dtype=int)
        result = fit._process_cv_groups(cv_groups=cv_groups, target=TARGET, classify=False)
        reference = pd.DataFrame({"index": np.arange(TARGET.shape[0]), "cv_group": cv_groups})
        pd.testing.assert_frame_equal(result, reference)

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
                solver_factr=1e7,
                chunks=1,
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
            solver_factr=None,
            chunks=1,
            num_jobs=None,
            random_state=None
        )
        self.assertEqual(len(result), 13)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        self.assertEqual(result["transform"], None)
        self.assertEqual(result["lambda_v_range"], fit.LAMBDA_V_RANGE)
        self.assertEqual(result["lambda_w_range"], fit.LAMBDA_W_RANGE)
        self.assertEqual(result["stage_1_trials"], 50)
        self.assertEqual(result["fit_mode"], fit.FitMode.BOTH)
        np.testing.assert_array_equal(result["num_batch_grid"], fit.NUM_BATCH_GRID)
        self.assertFalse(result["num_batch_grid"] is fit.NUM_BATCH_GRID)  # ensure mutable default is copied
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))  # classifier needs stratified splitter
        self.assertEqual(result["solver_factr"], fit.SOLVER_FACTR)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["chunker"], None)
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
            lambda_w_range=(1e-5, 1e-3),
            stage_1_trials=50,
            num_batch_grid=num_batch_grid,
            num_folds=5,
            solver_factr=(1e12, 1e10),
            chunks=2,
            num_jobs=2,
            random_state=12345
        )
        self.assertEqual(len(result), 13)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        self.assertTrue(isinstance(result["transform"], StandardScaler))
        self.assertFalse(result["transform"] is transform)
        self.assertEqual(result["lambda_v_range"], (1e-4, 1e-2))
        self.assertEqual(result["lambda_w_range"], (1e-5, 1e-3))
        self.assertEqual(result["stage_1_trials"], 50)
        self.assertEqual(result["fit_mode"], fit.FitMode.BOTH)
        np.testing.assert_array_equal(result["num_batch_grid"], num_batch_grid)
        self.assertFalse(result["num_batch_grid"] is num_batch_grid)
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))  # classifier needs stratified splitter
        self.assertEqual(result["solver_factr"], (1e12, 1e10))
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["chunker"], StratifiedKFold)
        self.assertEqual(result["num_jobs"], 2)
        self.assertTrue(isinstance(result["random_state"], np.random.RandomState))

    def test_process_select_settings_3(self):
        model = ClassifierModel()
        result = fit._process_select_settings(
            model=model,
            transform=None,
            lambda_v_range=1e-4,
            lambda_w_range=1e-5,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=5,
            solver_factr=1e10,
            chunks=1,
            num_jobs=None,
            random_state=None
        )
        self.assertEqual(len(result), 13)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        self.assertEqual(result["transform"], None)
        self.assertEqual(result["lambda_v_range"], 1e-4)
        self.assertEqual(result["lambda_w_range"], 1e-5)
        self.assertEqual(result["stage_1_trials"], 50)
        self.assertEqual(result["fit_mode"], fit.FitMode.NEITHER)
        np.testing.assert_array_equal(result["num_batch_grid"], fit.NUM_BATCH_GRID)
        self.assertFalse(result["num_batch_grid"] is fit.NUM_BATCH_GRID)  # ensure mutable default is copied
        self.assertTrue(isinstance(result["splitter"], StratifiedKFold))  # classifier needs stratified splitter
        self.assertEqual(result["solver_factr"], (1e10, 1e10))
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["chunker"], None)
        self.assertEqual(result["num_jobs"], None)
        self.assertTrue(isinstance(result["random_state"], np.random.RandomState))

    def test_check_select_settings_fail_1(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=-1.0,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
                solver_factr=(1e7, 1e7, 1e7),
                chunks=1,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must have length two if passing a tuple.")

    def test_check_select_settings_fail_10(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=(0.0, 1e7),
                chunks=1,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must be positive / have positive elements.")

    def test_check_select_settings_fail_11(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=(1e7, 0.0),
                chunks=1,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must be positive / have positive elements.")

    def test_check_select_settings_fail_12(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                chunks=1.0,
                num_jobs=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter chunks must be integer.")

    def test_check_select_settings_fail_13(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                chunks=0,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter chunks must be positive.")

    def test_check_select_settings_fail_14(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
                num_jobs=2.0
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_jobs must be integer if not passing None.")

    def test_check_select_settings_fail_15(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                chunks=1,
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
            solver_factr=fit.SOLVER_FACTR,
            chunks=1,
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

    # function fit._execute_stage_1() is already covered by tests for fit.select_hyperparameters()

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
            solver_factr=None,
            chunks=1,
            num_jobs=None,
            random_state=None
        )
        plan, lambdas = fit._make_stage_1_plan(
            settings=settings, features=FEATURES, target=TARGET, weights=WEIGHTS, cv_groups=None
        )
        self.assertEqual(len(plan), 3 * 50)
        self.assertEqual(len(plan[0]), 10)
        self.assertTrue(isinstance(plan[0]["model"], ClassifierModel))
        self.assertFalse(plan[0]["model"] is model)  # ensure original input is copied
        self.assertEqual(plan[0]["transform"], None)
        np.testing.assert_array_equal(plan[0]["features"], FEATURES)
        np.testing.assert_array_equal(plan[0]["target"], TARGET)
        np.testing.assert_array_equal(plan[0]["weights"], WEIGHTS)
        self.assertEqual(plan[0]["fold"], 0)
        self.assertEqual(plan[0]["train_ix"].shape, TARGET.shape)
        self.assertEqual(plan[0]["trial"], 0)
        self.assertEqual(len(plan[0]["parameters"]), 4)
        self.assertTrue(fit.LAMBDA_V_RANGE[0] <= plan[0]["parameters"]["lambda_v"] <= fit.LAMBDA_V_RANGE[1])
        self.assertTrue(fit.LAMBDA_W_RANGE[0] <= plan[0]["parameters"]["lambda_w"] <= fit.LAMBDA_W_RANGE[1])
        self.assertTrue(isinstance(plan[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(plan[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertEqual(plan[0]["chunker"], None)
        reference = [{
            "lambda_v": plan[i]["parameters"]["lambda_v"],
            "lambda_w": plan[i]["parameters"]["lambda_w"]
        } for i in range(50)]
        self.assertEqual(lambdas, reference)

    def test_make_stage_1_plan_2(self):
        model = ClassifierModel()
        transform = StandardScaler()
        settings = fit._process_select_settings(
            model=model,
            transform=transform,
            lambda_v_range=None,
            lambda_w_range=None,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=3,
            solver_factr=None,
            chunks=2,
            num_jobs=None,
            random_state=None
        )
        plan, lambdas = fit._make_stage_1_plan(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS,
            cv_groups=fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=True)
        )
        self.assertEqual(len(plan), 3 * 50)
        self.assertEqual(len(plan[0]), 10)
        self.assertTrue(isinstance(plan[0]["model"], ClassifierModel))
        self.assertFalse(plan[0]["model"] is model)  # ensure original input is copied
        self.assertTrue(isinstance(plan[0]["transform"], StandardScaler))
        self.assertFalse(plan[0]["transform"] is transform)  # ensure original input is copied
        np.testing.assert_allclose(
            plan[0]["transform"].mean_,
            np.mean(EXTENDED_FEATURES[plan[0]["train_ix"], :], axis=0),
            atol=1e-5
        )
        np.testing.assert_allclose(
            plan[0]["transform"].var_,
            np.var(EXTENDED_FEATURES[plan[0]["train_ix"], :], axis=0),
            atol=1e-5
        )
        np.testing.assert_array_equal(plan[0]["features"], EXTENDED_FEATURES)
        np.testing.assert_array_equal(plan[0]["target"], EXTENDED_TARGET)
        np.testing.assert_array_equal(plan[0]["weights"], EXTENDED_WEIGHTS)
        self.assertEqual(plan[0]["fold"], 0)
        self.assertEqual(plan[0]["train_ix"].shape, EXTENDED_TARGET.shape)
        self.assertEqual(plan[0]["trial"], 0)
        self.assertEqual(len(plan[0]["parameters"]), 4)
        self.assertTrue(fit.LAMBDA_V_RANGE[0] <= plan[0]["parameters"]["lambda_v"] <= fit.LAMBDA_V_RANGE[1])
        self.assertTrue(fit.LAMBDA_W_RANGE[0] <= plan[0]["parameters"]["lambda_w"] <= fit.LAMBDA_W_RANGE[1])
        self.assertTrue(isinstance(plan[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(plan[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertTrue(plan[0]["chunker"], StratifiedKFold)
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

    def test_make_train_ix_1(self):
        result = fit._make_train_ix(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, TARGET.shape)
        self.assertEqual(result[1].shape, TARGET.shape)
        self.assertEqual(result[2].shape, TARGET.shape)
        np.testing.assert_array_equal(
            result[0].astype(int) + result[1].astype(int) + result[2].astype(int),
            2 * np.ones(TARGET.shape[0], dtype=int)
        )  # every observation is included in two sets of training indices

    def test_make_train_ix_2(self):
        cv_groups = fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=True)
        result = fit._make_train_ix(
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            cv_groups=cv_groups,
            splitter=StratifiedKFold(n_splits=3)
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, EXTENDED_TARGET.shape)
        self.assertEqual(result[1].shape, EXTENDED_TARGET.shape)
        self.assertEqual(result[2].shape, EXTENDED_TARGET.shape)
        np.testing.assert_array_equal(
            result[0].astype(int) + result[1].astype(int) + result[2].astype(int),
            2 * np.ones(EXTENDED_TARGET.shape[0], dtype=int)
        )  # every observation is included in two sets of training indices
        for ix in result:  # each group is either completely contained in the training folds or validation fold
            train_groups = set(EXTENDED_CV_GROUPS[ix])
            validate_groups = set(EXTENDED_CV_GROUPS[np.logical_not(ix)])
            self.assertEqual(len(train_groups.intersection(validate_groups)), 0)

    def test_make_train_ix_3(self):
        cv_groups = fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=False)
        result = fit._make_train_ix(
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            cv_groups=cv_groups,
            splitter=KFold(n_splits=3)
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, EXTENDED_TARGET.shape)
        self.assertEqual(result[1].shape, EXTENDED_TARGET.shape)
        self.assertEqual(result[2].shape, EXTENDED_TARGET.shape)
        np.testing.assert_array_equal(
            result[0].astype(int) + result[1].astype(int) + result[2].astype(int),
            2 * np.ones(EXTENDED_TARGET.shape[0], dtype=int)
        )  # every observation is included in two sets of training indices
        for ix in result:  # each group is either completely contained in the training folds or validation fold
            train_groups = set(EXTENDED_CV_GROUPS[ix])
            validate_groups = set(EXTENDED_CV_GROUPS[np.logical_not(ix)])
            self.assertEqual(len(train_groups.intersection(validate_groups)), 0)

    def test_make_train_ix_4(self):
        features = np.vstack([FEATURES, FEATURES, FEATURES, FEATURES, FEATURES, FEATURES])
        target = np.hstack([TARGET, TARGET, TARGET, TARGET, TARGET, TARGET])
        cv_groups = fit._process_cv_groups(
            cv_groups=np.hstack([
                np.zeros(3 * TARGET.shape[0], dtype=int),
                np.ones(2 * TARGET.shape[0], dtype=int),
                2 * np.ones(TARGET.shape[0], dtype=int)
            ]),
            target=target,
            classify=True
        )
        message = ""
        warnings.filterwarnings("error")
        try:
            fit._make_train_ix(
                features=features, target=target, cv_groups=cv_groups, splitter=StratifiedKFold(n_splits=3)
            )
        except RuntimeWarning as ex:
            message = ex.args[0]
        warnings.resetwarnings()
        self.assertEqual(message, " ".join([
            "The quotient between the sizes of the largest and smallest cross-validation folds is greater than 1.20.",
            "This can happen if the number of groups defined in cv_groups is too small",
            "or the group sizes are too variable."
        ]))

    @staticmethod
    def test_binarize_index_1():
        result = fit._binarize_index(ix=np.array([1, 2, 4]), size=6)
        np.testing.assert_array_equal(result, np.array([False, True, True, False, True, False]))

    def test_make_transforms_1(self):
        train_ix = fit._make_train_ix(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )
        result = fit._make_transforms(features=FEATURES, train_ix=train_ix, transform=None)
        self.assertEqual(len(result), len(train_ix))
        self.assertEqual(result[0], None)

    def test_make_transforms_2(self):
        train_ix = fit._make_train_ix(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )
        transform = StandardScaler()
        result = fit._make_transforms(features=FEATURES, train_ix=train_ix, transform=transform)
        self.assertEqual(len(result), len(train_ix))
        self.assertTrue(isinstance(result[0], StandardScaler))
        self.assertFalse(result[0] is transform)  # ensure original input is copied
        np.testing.assert_allclose(result[0].mean_, np.mean(FEATURES[train_ix[0], :], axis=0), atol=1e-5)
        np.testing.assert_allclose(result[0].var_, np.var(FEATURES[train_ix[0], :], axis=0), atol=1e-5)

    # function fit._execute_plan() is already covered by tests for fit.select_hyperparameters()

    # function fit._fit_stage_1() is already covered by tests for fit.select_hyperparameters()

    @staticmethod
    def test_prepare_features_1():
        train_ix = fit._make_train_ix(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )[0]
        result = fit._prepare_features(features=FEATURES, sample_ix=train_ix, transform=None)
        np.testing.assert_array_equal(result, FEATURES[train_ix, :])

    @staticmethod
    def test_prepare_features_2():
        transform = StandardScaler().fit(FEATURES)
        train_ix = fit._make_train_ix(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )[0]
        result = fit._prepare_features(features=FEATURES, sample_ix=train_ix, transform=transform)
        np.testing.assert_allclose(result, transform.transform(FEATURES[train_ix, :]), atol=1e-5)

    # function fit._fit_model() is already covered by tests for fit.select_hyperparameters()

    def test_get_first_chunk(self):
        target = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        train_ix = np.array([True, True, False, True, True, False, True, True, False])
        use_target, use_ix = fit._get_first_chunk(
            target=target,
            train_ix=train_ix,
            chunker=StratifiedKFold(n_splits=2, shuffle=True, random_state=np.random.RandomState(12345))
        )
        np.testing.assert_array_equal(use_target, np.array([0, 1, 2]))
        self.assertEqual(use_ix.shape, (9,))
        self.assertEqual(np.nonzero(use_ix)[0].shape, (3, ))
        np.testing.assert_array_equal(np.logical_and(train_ix, use_ix), use_ix)

    # function fit._update_index() already tested by the above

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

    # function fit._execute_stage_2() is already covered by tests for fit.select_hyperparameters()

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
            solver_factr=None,
            chunks=1,
            num_jobs=None,
            random_state=None
        )
        result = fit._make_stage_2_plan(
            settings=settings, features=FEATURES, target=TARGET, weights=WEIGHTS, cv_groups=None
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 9)
        self.assertTrue(isinstance(result[0]["model"], ClassifierModel))
        self.assertFalse(result[0]["model"] is model)  # ensure original input is copied
        self.assertEqual(result[0]["transform"], None)
        np.testing.assert_array_equal(result[0]["features"], FEATURES)
        np.testing.assert_array_equal(result[0]["target"], TARGET)
        np.testing.assert_array_equal(result[0]["weights"], WEIGHTS)
        np.testing.assert_array_equal(result[0]["num_batch_grid"], fit.NUM_BATCH_GRID)
        self.assertEqual(result[0]["train_ix"].shape, TARGET.shape)
        self.assertEqual(len(result[0]["parameters"]), 3)
        self.assertEqual(result[0]["parameters"]["n_iter"], fit.NUM_BATCH_GRID[-1])
        self.assertTrue(isinstance(result[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(result[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertEqual(result[0]["chunker"], None)

    def test_make_stage_2_plan_2(self):
        model = ClassifierModel()
        transform = StandardScaler()
        settings = fit._process_select_settings(
            model=model,
            transform=transform,
            lambda_v_range=None,
            lambda_w_range=None,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=3,
            solver_factr=None,
            chunks=2,
            num_jobs=None,
            random_state=None
        )
        result = fit._make_stage_2_plan(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS,
            cv_groups=fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=True)
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 9)
        self.assertTrue(isinstance(result[0]["model"], ClassifierModel))
        self.assertFalse(result[0]["model"] is model)  # ensure original input is copied
        self.assertTrue(isinstance(result[0]["transform"], StandardScaler))
        self.assertFalse(result[0]["transform"] is transform)  # ensure original input is copied
        np.testing.assert_allclose(
            result[0]["transform"].mean_,
            np.mean(EXTENDED_FEATURES[result[0]["train_ix"], :], axis=0),
            atol=1e-5
        )
        np.testing.assert_allclose(
            result[0]["transform"].var_,
            np.var(EXTENDED_FEATURES[result[0]["train_ix"], :], axis=0),
            atol=1e-5
        )
        np.testing.assert_array_equal(result[0]["features"], EXTENDED_FEATURES)
        np.testing.assert_array_equal(result[0]["target"], EXTENDED_TARGET)
        np.testing.assert_array_equal(result[0]["weights"], EXTENDED_WEIGHTS)
        np.testing.assert_array_equal(result[0]["num_batch_grid"], fit.NUM_BATCH_GRID)
        self.assertEqual(result[0]["train_ix"].shape, EXTENDED_TARGET.shape)
        self.assertEqual(len(result[0]["parameters"]), 3)
        self.assertEqual(result[0]["parameters"]["n_iter"], fit.NUM_BATCH_GRID[-1])
        self.assertTrue(isinstance(result[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(result[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertTrue(isinstance(result[0]["chunker"], StratifiedKFold))

    # function fit._fit_stage_2() is already covered by tests for fit.select_hyperparameters()

    # function fit._fit_model_chunked() is already covered by tests for fit.select_hyperparameters()

    # function fit._add_chunks() is already covered by tests for fit.select_hyperparameters()

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

    def test_make_final_model_1(self):
        settings = fit._process_select_settings(
            model=ClassifierModel(n_iter=2),
            transform=StandardScaler(),
            lambda_v_range=1e-4,
            lambda_w_range=1e-5,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=5,
            solver_factr=1e10,
            chunks=1,
            num_jobs=None,
            random_state=None
        )
        model, chunk_ix = fit._make_final_model(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS
        )
        self.assertTrue(isinstance(model, Pipeline))
        self.assertTrue(isinstance(model["transform"], StandardScaler))
        self.assertTrue(isinstance(model["model"], ClassifierModel))
        check_is_fitted(model)
        self.assertEqual(chunk_ix, None)

    def test_make_final_model_2(self):
        settings = fit._process_select_settings(
            model=ClassifierModel(n_iter=2),
            transform=None,
            lambda_v_range=1e-4,
            lambda_w_range=1e-5,
            stage_1_trials=50,
            num_batch_grid=None,
            num_folds=5,
            solver_factr=1e10,
            chunks=2,
            num_jobs=None,
            random_state=None
        )
        model, chunk_ix = fit._make_final_model(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS
        )
        self.assertTrue(isinstance(model, ClassifierModel))
        check_is_fitted(model)
        self.assertEqual(len(chunk_ix), 2)
        chunk_1 = set(chunk_ix[0])
        chunk_2 = set(chunk_ix[1])
        self.assertEqual(len(chunk_1.intersection(chunk_2)), 0)
        self.assertEqual(len(chunk_1.union(chunk_2)), EXTENDED_TARGET.shape[0])

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
