"""Unit tests for code in the utility.fit submodule.

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
import proset.utility.fit as fit
from test.test_np_objective import FEATURES, TARGET, WEIGHTS  # pylint: disable=wrong-import-order


EXTENDED_FEATURES = np.vstack([FEATURES, FEATURES, FEATURES])
# sample data is too small for some tests involving cross-validation
EXTENDED_TARGET = np.hstack([TARGET, TARGET, TARGET])
EXTENDED_WEIGHTS = np.hstack([WEIGHTS, WEIGHTS, WEIGHTS])
EXTENDED_CV_GROUPS = np.hstack([np.zeros_like(TARGET), np.ones_like(TARGET), 2 * np.ones_like(TARGET)])
TRAIN_SIZE = (int(np.floor(TARGET.shape[0] * 2.0 / 3.0)), int(np.ceil(TARGET.shape[0] * 2.0 / 3.0)))
# bounds on combined training fold size for 3 CV folds
MAX_SAMPLES = int(np.ceil(EXTENDED_TARGET.shape[0] / 2))
EXTENDED_TRAIN_SIZE = (
    int(np.floor(EXTENDED_TARGET.shape[0] * 2.0 / 3.0)), int(np.ceil(EXTENDED_TARGET.shape[0] * 2.0 / 3.0))
)


# pylint: disable=missing-function-docstring, protected-access, too-many-public-methods
class TestUtility(TestCase):
    """Unit tests for selected helper functions in proset.utility.
    """

    def test_subsampler_init_fail_1(self):
        message = ""
        try:
            fit.Subsampler(subsample_rate=0.0, stratify=False, random_state=12345)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter subsample_rate must lie in (0.0, 1.0).")

    def test_subsampler_init_fail_2(self):
        message = ""
        try:
            fit.Subsampler(subsample_rate=1.0, stratify=False, random_state=12345)
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter subsample_rate must lie in (0.0, 1.0).")

    # correct working of Subsampler.__init__() tested by method tests below

    def test_subsampler_subsample_1(self):
        y = np.hstack([np.ones(5), np.zeros(4)])
        subsampler = fit.Subsampler(subsample_rate=0.5, stratify=False, random_state=12345)
        ix = subsampler.subsample(y)
        self.assertEqual(ix.shape[0], 5)  # subsampler rounds up
        self.assertTrue(np.min(ix) >= 0)
        self.assertTrue(np.max(ix) < EXTENDED_TARGET.shape[0])
        np.testing.assert_array_equal(ix, np.sort(ix))

    def test_subsampler_subsample_2(self):
        y = np.hstack([np.ones(5), np.zeros(3)])
        subsampler = fit.Subsampler(subsample_rate=0.5, stratify=True, random_state=12345)
        ix = subsampler.subsample(y)
        self.assertEqual(ix.shape[0], 5)  # stratified subsampler rounds up per class
        self.assertTrue(np.min(ix) >= 0)
        self.assertTrue(np.max(ix) < EXTENDED_TARGET.shape[0])
        np.testing.assert_array_equal(ix, np.sort(ix))
        self.assertEqual(np.sum(y[ix] == 1), 3)
        self.assertEqual(np.sum(y[ix] == 0), 2)

    # Subsampler._create_sample_index() already tested by the above

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
            max_samples=MAX_SAMPLES,
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
        self.assertEqual(len(result["sample_ix"]), result["model"]["model"].set_manager_.num_batches)

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
                max_samples=None,
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
            max_samples=None,
            num_jobs=None,
            random_state=None
        )
        self.assertEqual(len(result), 12)
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
        self.assertEqual(result["max_samples"], None)
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
            max_samples=MAX_SAMPLES,
            num_jobs=2,
            random_state=12345
        )
        self.assertEqual(len(result), 12)
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
        self.assertEqual(result["max_samples"], MAX_SAMPLES)
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
            max_samples=None,
            num_jobs=None,
            random_state=None
        )
        self.assertEqual(len(result), 12)
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
        self.assertEqual(result["max_samples"], None)
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=None,
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
                max_samples=1.0,
                num_jobs=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_samples must be integer if not passing None.")

    def test_check_select_settings_fail_13(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                max_samples=0,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_samples must be positive if not passing None.")

    def test_check_select_settings_fail_14(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_range=fit.LAMBDA_V_RANGE,
                lambda_w_range=fit.LAMBDA_W_RANGE,
                stage_1_trials=50,
                num_batch_grid=fit.NUM_BATCH_GRID,
                solver_factr=fit.SOLVER_FACTR,
                max_samples=None,
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
                max_samples=None,
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
            max_samples=None,
            num_jobs=None
        )

    @staticmethod
    def test_check_select_settings_2():
        fit._check_select_settings(
            lambda_v_range=fit.LAMBDA_V_RANGE,
            lambda_w_range=fit.LAMBDA_W_RANGE,
            stage_1_trials=50,
            num_batch_grid=fit.NUM_BATCH_GRID,
            solver_factr=fit.SOLVER_FACTR,
            max_samples=1000,
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
            max_samples=None,
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
        self.assertTrue(TRAIN_SIZE[0] <= plan[0]["train_ix"].shape[0] <= TRAIN_SIZE[1])
        self.assertEqual(plan[0]["trial"], 0)
        self.assertEqual(len(plan[0]["parameters"]), 4)
        self.assertTrue(fit.LAMBDA_V_RANGE[0] <= plan[0]["parameters"]["lambda_v"] <= fit.LAMBDA_V_RANGE[1])
        self.assertTrue(fit.LAMBDA_W_RANGE[0] <= plan[0]["parameters"]["lambda_w"] <= fit.LAMBDA_W_RANGE[1])
        self.assertTrue(isinstance(plan[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(plan[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertEqual(plan[0]["subsampler"], None)
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
            max_samples=MAX_SAMPLES,
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
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= plan[0]["train_ix"].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        self.assertEqual(plan[0]["trial"], 0)
        self.assertEqual(len(plan[0]["parameters"]), 4)
        self.assertTrue(fit.LAMBDA_V_RANGE[0] <= plan[0]["parameters"]["lambda_v"] <= fit.LAMBDA_V_RANGE[1])
        self.assertTrue(fit.LAMBDA_W_RANGE[0] <= plan[0]["parameters"]["lambda_w"] <= fit.LAMBDA_W_RANGE[1])
        self.assertTrue(isinstance(plan[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(plan[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertTrue(isinstance(plan[0]["subsampler"], fit.Subsampler))
        self.assertTrue(plan[0]["subsampler"]._stratify)
        reference = [{
            "lambda_v": plan[i]["parameters"]["lambda_v"],
            "lambda_w": plan[i]["parameters"]["lambda_w"]
        } for i in range(50)]
        self.assertEqual(lambdas, reference)

    def test_make_stage_1_plan_3(self):
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
            max_samples=FEATURES.shape[0],  # no subsampling required
            num_jobs=None,
            random_state=None
        )
        plan, _ = fit._make_stage_1_plan(
            settings=settings, features=FEATURES, target=TARGET, weights=WEIGHTS, cv_groups=None
        )
        self.assertEqual(plan[0]["subsampler"], None)  # check only difference to above

    def test_sample_lambda_1(self):
        result = fit._sample_lambda(
            lambda_range=1e-3,
            trials=10,
            randomize=False,
            random_state=np.random.RandomState()
        )
        self.assertEqual(result.shape, (10, ))
        self.assertTrue(np.all(result == 1e-3))

    def test_sample_lambda_2(self):
        result = fit._sample_lambda(
            lambda_range=fit.LAMBDA_V_RANGE,
            trials=10,
            randomize=False,
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
            randomize=True,
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
        self.assertTrue(TRAIN_SIZE[0] <= result[0].shape[0] <= TRAIN_SIZE[1])
        self.assertTrue(TRAIN_SIZE[0] <= result[1].shape[0] <= TRAIN_SIZE[1])
        self.assertTrue(TRAIN_SIZE[0] <= result[2].shape[0] <= TRAIN_SIZE[1])
        values, counts = np.unique(np.hstack(result), return_counts=True)
        np.testing.assert_array_equal(values, np.arange(TARGET.shape[0]))
        self.assertTrue(np.all(counts == 2))  # every observation is included in two sets of training indices

    def test_make_train_ix_2(self):
        cv_groups = fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=True)
        result = fit._make_train_ix(
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            cv_groups=cv_groups,
            splitter=StratifiedKFold(n_splits=3)
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[0].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[1].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[2].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        values, counts = np.unique(np.hstack(result), return_counts=True)
        np.testing.assert_array_equal(values, np.arange(EXTENDED_TARGET.shape[0]))
        self.assertTrue(np.all(counts == 2))  # every observation is included in two sets of training indices
        for ix in result:  # each group is either completely contained in the training folds or validation fold
            train_groups = set(EXTENDED_CV_GROUPS[ix])
            validate_groups = set(EXTENDED_CV_GROUPS[fit._invert_index(index=ix, max_value=EXTENDED_TARGET.shape[0])])
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
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[0].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[1].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[2].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        values, counts = np.unique(np.hstack(result), return_counts=True)
        np.testing.assert_array_equal(values, np.arange(EXTENDED_TARGET.shape[0]))
        self.assertTrue(np.all(counts == 2))  # every observation is included in two sets of training indices
        for ix in result:  # each group is either completely contained in the training folds or validation fold
            train_groups = set(EXTENDED_CV_GROUPS[ix])
            validate_groups = set(EXTENDED_CV_GROUPS[fit._invert_index(index=ix, max_value=EXTENDED_TARGET.shape[0])])
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

    def test_make_subsampler_1(self):
        subsampler = fit._make_subsampler(max_samples=None, num_samples=10, stratify=True, random_state=12345)
        self.assertEqual(subsampler, None)

    def test_make_subsampler_2(self):
        subsampler = fit._make_subsampler(max_samples=10, num_samples=10, stratify=True, random_state=12345)
        self.assertEqual(subsampler, None)

    def test_make_subsampler_3(self):
        subsampler = fit._make_subsampler(max_samples=5, num_samples=10, stratify=True, random_state=12345)
        self.assertTrue(isinstance(subsampler, fit.Subsampler))
        self.assertEqual(subsampler._subsample_rate, 0.5)
        self.assertTrue(subsampler._stratify)

    def test_make_subsampler_4(self):
        subsampler = fit._make_subsampler(max_samples=3, num_samples=10, stratify=False, random_state=12345)
        self.assertTrue(isinstance(subsampler, fit.Subsampler))
        self.assertEqual(subsampler._subsample_rate, 0.3)
        self.assertFalse(subsampler._stratify)

    # functions fit._execute_plan(), fit._fit_stage_1(), and fit._fit_model() are already covered by tests for
    # fit.select_hyperparameters()

    @staticmethod
    def test_invert_index_1():
        complement = fit._invert_index(index=np.array([1, 3, 5]), max_value=6)
        np.testing.assert_array_equal(complement, np.array([0, 2, 4]))

    @staticmethod
    def test_get_training_samples_1():
        train_ix = np.arange(0, TARGET.shape[0], 2)
        target, use_ix = fit._get_training_samples(target=TARGET, train_ix=train_ix, subsampler=None)
        np.testing.assert_array_equal(target, TARGET[train_ix])
        np.testing.assert_array_equal(use_ix, train_ix)

    def test_get_training_samples_2(self):
        train_ix = np.arange(0, TARGET.shape[0], 2)
        subsampler = fit.Subsampler(subsample_rate=0.5, stratify=False, random_state=12345)
        target, use_ix = fit._get_training_samples(target=TARGET, train_ix=train_ix, subsampler=subsampler)
        reduced_length = int(np.ceil(train_ix.shape[0] * 0.5))
        self.assertEqual(target.shape[0], reduced_length)
        self.assertEqual(use_ix.shape[0], reduced_length)
        self.assertEqual(set(use_ix).intersection(set(train_ix)), set(use_ix))

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
            max_samples=None,
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
        self.assertTrue(TRAIN_SIZE[0] <= result[0]["train_ix"].shape[0] <= TRAIN_SIZE[1])
        self.assertEqual(len(result[0]["parameters"]), 3)
        self.assertEqual(result[0]["parameters"]["n_iter"], fit.NUM_BATCH_GRID[-1])
        self.assertTrue(isinstance(result[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(result[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertEqual(result[0]["subsampler"], None)

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
            max_samples=MAX_SAMPLES,
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
        self.assertTrue(EXTENDED_TRAIN_SIZE[0] <= result[0]["train_ix"].shape[0] <= EXTENDED_TRAIN_SIZE[1])
        self.assertEqual(len(result[0]["parameters"]), 3)
        self.assertEqual(result[0]["parameters"]["n_iter"], fit.NUM_BATCH_GRID[-1])
        self.assertTrue(isinstance(result[0]["parameters"]["random_state"], np.random.RandomState))
        self.assertEqual(result[0]["parameters"]["solver_factr"], fit.SOLVER_FACTR[0])
        self.assertTrue(isinstance(result[0]["subsampler"], fit.Subsampler))
        self.assertTrue(result[0]["subsampler"]._stratify)

    def test_make_stage_2_plan_3(self):
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
            max_samples=FEATURES.shape[0],
            num_jobs=None,
            random_state=None
        )
        result = fit._make_stage_2_plan(
            settings=settings, features=FEATURES, target=TARGET, weights=WEIGHTS, cv_groups=None
        )
        self.assertEqual(result[0]["subsampler"], None)  # check only difference to above

    # function fit._fit_stage_2(), fit._fit_with_subsampling(), and fit._use_subsampling() are already covered by tests
    # for fit.select_hyperparameters()

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
            max_samples=None,
            num_jobs=None,
            random_state=None
        )
        model, sample_ix = fit._make_final_model(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS
        )
        self.assertTrue(isinstance(model, Pipeline))
        self.assertTrue(isinstance(model["transform"], StandardScaler))
        self.assertTrue(isinstance(model["model"], ClassifierModel))
        check_is_fitted(model)
        self.assertEqual(sample_ix, None)

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
            max_samples=MAX_SAMPLES,
            num_jobs=None,
            random_state=None
        )
        model, sample_ix = fit._make_final_model(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS
        )
        self.assertTrue(isinstance(model, ClassifierModel))
        check_is_fitted(model)
        self.assertEqual(len(sample_ix), 2)
        self.assertTrue(MAX_SAMPLES <= sample_ix[0].shape[0] <= MAX_SAMPLES + 1)
        # stratified sampling on two classes can at most generate one more sample than specified due to rounding errors
        self.assertTrue(MAX_SAMPLES <= sample_ix[1].shape[0] <= MAX_SAMPLES + 1)
