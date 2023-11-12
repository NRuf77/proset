"""Unit tests for code in the utility.fit submodule.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from unittest import TestCase
import warnings

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from proset import ClassifierModel
from proset.utility import fit
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
LAMBDA_V_GRID = np.logspace(-4.0, -1.0, 7)
MAX_BATCHES = 2
NUM_FOLDS = 3
SOLVER_FACTR = (1e10, 1e8)
RANDOM_STATE = 12345


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

    # correct behavior of Subsampler.__init__() on valid input is verified by tests for Subsampler.subsample() below

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

    # function Subsampler._create_sample_index() is already covered by the tests for Subsampler.subsample() above

    def test_select_hyperparameters_fail_1(self):
        message = ""
        try:  # trigger one error from _check_select_input() to ensure it is called
            fit.select_hyperparameters(
                model=ClassifierModel(),
                features=FEATURES,
                target=TARGET,
                cv_groups=np.ones((TARGET.shape[0], 1), dtype=int)
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter cv_groups must be a 1D array.")

    def test_select_hyperparameters_1(self):
        model = ClassifierModel()
        result = fit.select_hyperparameters(model=model, features=EXTENDED_FEATURES, target=EXTENDED_TARGET)
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result["model"], ClassifierModel))
        self.assertFalse(result["model"] is model)  # ensure original input is copied
        self.assertFalse(result["model"].random_state is None)  # ensure random state is explicitly managed
        check_is_fitted(result["model"])
        self.assertEqual(sorted(list(result["search"].keys())), ["best_ix", "cv_results", "selected_ix", "threshold"])
        # benchmark cases serve as integration test where results can be reviewed in detail

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
            lambda_v_grid=LAMBDA_V_GRID,
            max_batches=MAX_BATCHES,
            num_folds=NUM_FOLDS,
            solver_factr=SOLVER_FACTR,
            max_samples=MAX_SAMPLES,
            num_jobs=None,
            random_state=RANDOM_STATE
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
        self.assertFalse(result["model"]["model"].random_state is None)  # ensure random state is explicitly managed
        self.assertEqual(sorted(list(result["search"].keys())), ["best_ix", "cv_results", "selected_ix", "threshold"])
        self.assertEqual(len(result["sample_ix"]), result["model"]["model"].set_manager_.num_batches)
        # benchmark cases serve as integration test where results can be reviewed in detail

    def test_select_hyperparameters_3(self):
        model = ClassifierModel()
        transform = StandardScaler()
        result = fit.select_hyperparameters(
            model=model,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            transform=transform,
            lambda_v_grid=1e-3,
            num_jobs=2
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result["model"], Pipeline))
        check_is_fitted(result["model"])
        self.assertTrue(isinstance(result["model"]["transform"], StandardScaler))
        self.assertFalse(result["model"]["transform"] is transform)  # ensure original input is copied
        np.testing.assert_allclose(result["model"]["transform"].mean_, np.mean(EXTENDED_FEATURES, axis=0), atol=1e-6)
        np.testing.assert_allclose(result["model"]["transform"].var_, np.var(EXTENDED_FEATURES, axis=0), atol=1e-6)
        self.assertTrue(isinstance(result["model"]["model"], ClassifierModel))
        self.assertFalse(result["model"]["model"] is model)
        self.assertFalse(result["model"]["model"].random_state is None)  # ensure random state is explicitly managed
        self.assertEqual(sorted(list(result["search"].keys())), ["best_ix", "cv_results", "selected_ix", "threshold"])
        # benchmark cases serve as integration test where results can be reviewed in detail

    def test_check_select_input_fail_1(self):
        message = ""
        try:  # trigger one error from _process_cv_groups() to ensure it is called
            fit._check_select_input(
                model=ClassifierModel(),
                target=TARGET,
                cv_groups=np.ones((TARGET.shape[0], 1), dtype=int),
                transform=None,
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                num_folds=NUM_FOLDS,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter cv_groups must be a 1D array.")

    def test_check_select_input_fail_2(self):
        message = ""
        try:  # trigger one error from _process_select_settings() to ensure it is called
            fit._check_select_input(
                model=ClassifierModel(),
                target=TARGET,
                cv_groups=None,
                transform=None,
                lambda_v_grid=np.array([[1e-3]]),
                max_batches=MAX_BATCHES,
                num_folds=NUM_FOLDS,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v_grid must be a 1D array.")

    def test_check_select_input_1(self):
        model = ClassifierModel()
        target, cv_groups, settings = fit._check_select_input(
            model=ClassifierModel(),
            target=TARGET,
            cv_groups=None,
            transform=None,
            lambda_v_grid=LAMBDA_V_GRID,
            max_batches=MAX_BATCHES,
            num_folds=NUM_FOLDS,
            solver_factr=SOLVER_FACTR,
            max_samples=MAX_SAMPLES,
            num_jobs=None,
            random_state=RANDOM_STATE
        )
        np.testing.assert_array_equal(target, TARGET)
        self.assertEqual(cv_groups, None)
        self.assertEqual(len(settings), 9)
        self.assertTrue(isinstance(settings["model"], ClassifierModel))
        self.assertFalse(settings["model"] is model)  # ensure original input is copied
        self.assertEqual(settings["transform"], None)
        np.testing.assert_array_equal(settings["lambda_v_grid"], LAMBDA_V_GRID)
        self.assertEqual(settings["max_batches"], MAX_BATCHES)
        self.assertTrue(isinstance(settings["splitter"], StratifiedKFold))
        self.assertEqual(settings["splitter"].n_splits, NUM_FOLDS)
        self.assertEqual(settings["solver_factr"], SOLVER_FACTR)
        self.assertEqual(settings["max_samples"], MAX_SAMPLES)
        self.assertEqual(settings["num_jobs"], None)
        self.assertTrue(isinstance(settings["random_state"], np.random.RandomState))

    def test_check_select_input_2(self):
        model = ClassifierModel()
        transform = StandardScaler()
        target, cv_groups, settings = fit._check_select_input(
            model=ClassifierModel(),
            target=list(EXTENDED_TARGET),
            cv_groups=EXTENDED_CV_GROUPS,
            transform=transform,
            lambda_v_grid=1e-3,
            max_batches=MAX_BATCHES,
            num_folds=NUM_FOLDS,
            solver_factr=1e7,
            max_samples=MAX_SAMPLES,
            num_jobs=None,
            random_state=RANDOM_STATE
        )
        np.testing.assert_array_equal(target, EXTENDED_TARGET)
        self.assertEqual(list(cv_groups.columns), ["index", "cv_group", "target"])
        self.assertEqual(cv_groups.shape[0], target.shape[0])  # content tested below for  _process_cv_groups()
        self.assertEqual(len(settings), 9)
        self.assertTrue(isinstance(settings["model"], ClassifierModel))
        self.assertFalse(settings["model"] is model)  # ensure original input is copied
        self.assertTrue(isinstance(settings["transform"], StandardScaler))
        self.assertFalse(settings["transform"] is transform)
        np.testing.assert_array_equal(settings["lambda_v_grid"], np.array([1e-3]))
        self.assertEqual(settings["max_batches"], MAX_BATCHES)
        self.assertTrue(isinstance(settings["splitter"], StratifiedKFold))
        self.assertEqual(settings["splitter"].n_splits, NUM_FOLDS)
        self.assertEqual(settings["solver_factr"], (1e7, 1e7))
        self.assertEqual(settings["max_samples"], MAX_SAMPLES)
        self.assertEqual(settings["num_jobs"], None)
        self.assertTrue(isinstance(settings["random_state"], np.random.RandomState))

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
            fit._process_cv_groups(cv_groups=np.ones(TARGET.shape[0] + 1, dtype=int), target=TARGET, classify=True)
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

    def test_process_cv_groups_2(self):
        result = fit._process_cv_groups(cv_groups=np.ones(TARGET.shape[0], dtype=int), target=TARGET, classify=True)
        self.assertEqual(result.shape, (TARGET.shape[0], 3))
        np.testing.assert_array_equal(result["index"].values, np.arange(TARGET.shape[0]))
        np.testing.assert_array_equal(result["cv_group"].values, np.ones(TARGET.shape[0]))
        np.testing.assert_array_equal(result["target"].values, TARGET)

    def test_process_cv_groups_3(self):
        result = fit._process_cv_groups(cv_groups=np.ones(TARGET.shape[0], dtype=int), target=TARGET, classify=False)
        self.assertEqual(result.shape, (TARGET.shape[0], 2))
        np.testing.assert_array_equal(result["index"].values, np.arange(TARGET.shape[0]))
        np.testing.assert_array_equal(result["cv_group"].values, np.ones(TARGET.shape[0]))

    def test_process_select_settings_fail_1(self):
        message = ""
        try:  # trigger one error from _check_select_settings() to ensure it is called
            fit._process_select_settings(
                model=ClassifierModel(),
                transform=None,
                lambda_v_grid=np.array([[1e-3]]),
                max_batches=MAX_BATCHES,
                num_folds=NUM_FOLDS,
                solver_factr=SOLVER_FACTR,
                num_jobs=None,
                max_samples=MAX_SAMPLES,
                random_state=RANDOM_STATE
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v_grid must be a 1D array.")

    # function _process_select_settings() is already covered by the tests for _check_select_input() above

    def test_check_select_settings_fail_1(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=np.array([[1e-3]]),
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v_grid must be a 1D array.")

    def test_check_select_settings_fail_2(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=np.array([-1e-3, 1e-3]),
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v_grid must not contain negative values.")

    def test_check_select_settings_fail_3(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=np.array([1e-2, 1e-3]),
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter lambda_v_grid must contain values in ascending order.")

    def test_check_select_settings_fail_4(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=1.0,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_batches must be integer.")

    def test_check_select_settings_fail_5(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=0,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_batches must be positive.")

    def test_check_select_settings_fail_6(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=(1e7, 1e7, 1e7),
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must have length two if passing a tuple.")

    def test_check_select_settings_fail_7(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=(0.0, 1e7),
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must be positive / have positive elements.")

    def test_check_select_settings_fail_8(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=(1e7, 0.0),
                max_samples=MAX_SAMPLES,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter solver_factr must be positive / have positive elements.")

    def test_check_select_settings_fail_9(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=1.0,
                num_jobs=None
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_samples must be integer if not passing None.")

    def test_check_select_settings_fail_10(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=0,
                num_jobs=None
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter max_samples must be positive if not passing None.")

    def test_check_select_settings_fail_11(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=1.0
            )
        except TypeError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_jobs must be integer if not passing None.")

    def test_check_select_settings_fail_12(self):
        message = ""
        try:
            fit._check_select_settings(
                lambda_v_grid=LAMBDA_V_GRID,
                max_batches=MAX_BATCHES,
                solver_factr=SOLVER_FACTR,
                max_samples=MAX_SAMPLES,
                num_jobs=1
            )
        except ValueError as ex:
            message = ex.args[0]
        self.assertEqual(message, "Parameter num_jobs must be greater than 1 if not passing None.")

    # behavior of _check_select_settings() on correct input is already verified by tests for _check_select_input() above

    # function _execute_search() is already covered by the tests for select_hyperparameters() above

    def test_make_plan_1(self):
        model = ClassifierModel()
        settings = fit._process_select_settings(
            model=model,
            transform=None,
            lambda_v_grid=LAMBDA_V_GRID,
            max_batches=MAX_BATCHES,
            num_folds=3,
            solver_factr=SOLVER_FACTR,
            num_jobs=None,
            max_samples=MAX_SAMPLES,
            random_state=RANDOM_STATE
        )
        result = fit._make_plan(settings=settings, features=FEATURES, target=TARGET, weights=None, cv_groups=None)
        self.assertEqual(len(result), LAMBDA_V_GRID.shape[0] * NUM_FOLDS)
        self.assertEqual(len(result[0]), 9)
        self.assertTrue(isinstance(result[0]["model"], ClassifierModel))
        self.assertFalse(result[0]["model"] is model)  # ensure original input is copied
        self.assertEqual(result[0]["transform"], None)
        np.testing.assert_array_equal(result[0]["features"], FEATURES)
        np.testing.assert_array_equal(result[0]["target"], TARGET)
        self.assertEqual(result[0]["weights"], None)
        self.assertEqual(result[0]["trial"], 0)
        self.assertEqual(result[0]["fold"], 0)
        self.assertTrue(TRAIN_SIZE[0] <= result[0]["train_ix"].shape[0] <= TRAIN_SIZE[1])
        self.assertEqual(result[0]["subsampler"], None)
        self.assertEqual(result[-1]["trial"], LAMBDA_V_GRID.shape[0] - 1)
        self.assertEqual(result[-1]["fold"], 2)

    def test_make_plan_2(self):
        model = ClassifierModel()
        transform = StandardScaler()
        settings = fit._process_select_settings(
            model=model,
            transform=transform,
            lambda_v_grid=LAMBDA_V_GRID,
            max_batches=MAX_BATCHES,
            num_folds=3,
            solver_factr=SOLVER_FACTR,
            num_jobs=None,
            max_samples=MAX_SAMPLES,
            random_state=RANDOM_STATE
        )
        cv_groups = fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=True)
        result = fit._make_plan(
            settings=settings,
            features=EXTENDED_FEATURES,
            target=EXTENDED_TARGET,
            weights=EXTENDED_WEIGHTS,
            cv_groups=cv_groups
        )
        self.assertEqual(len(result), LAMBDA_V_GRID.shape[0] * NUM_FOLDS)
        self.assertEqual(len(result[0]), 9)
        self.assertTrue(isinstance(result[0]["model"], ClassifierModel))
        self.assertFalse(result[0]["model"] is model)  # ensure original input is copied
        self.assertTrue(isinstance(result[0]["transform"], StandardScaler))
        self.assertFalse(result[0]["transform"] is transform)
        np.testing.assert_allclose(
            result[0]["transform"].mean_, np.mean(EXTENDED_FEATURES[result[0]["train_ix"], :], axis=0), atol=1e-5
        )
        np.testing.assert_allclose(
            result[0]["transform"].var_, np.var(EXTENDED_FEATURES[result[0]["train_ix"], :], axis=0), atol=1e-5
        )
        np.testing.assert_array_equal(result[0]["features"], EXTENDED_FEATURES)
        np.testing.assert_array_equal(result[0]["target"], EXTENDED_TARGET)
        np.testing.assert_array_equal(result[0]["weights"], EXTENDED_WEIGHTS)
        self.assertEqual(result[0]["trial"], 0)
        self.assertEqual(result[0]["fold"], 0)
        train_size = (
            int(np.floor(EXTENDED_TARGET.shape[0] * 2.0 / 3.0)),
            int(np.ceil(EXTENDED_TARGET.shape[0] * 2.0 / 3.0))
        )
        self.assertTrue(train_size[0] <= result[0]["train_ix"].shape[0] <= train_size[1])
        self.assertTrue(isinstance(result[0]["subsampler"], fit.Subsampler))
        self.assertTrue(result[0]["subsampler"]._stratify)
        self.assertEqual(result[-1]["trial"], LAMBDA_V_GRID.shape[0] - 1)
        self.assertEqual(result[-1]["fold"], 2)

    def test_sample_random_state_1(self):
        random_state = np.random.RandomState(12345)
        result = fit._sample_random_state(random_state)
        self.assertTrue(isinstance(result, np.random.RandomState))
        self.assertFalse(result is random_state)

    def test_make_train_folds_1(self):
        result = fit._make_train_folds(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(TRAIN_SIZE[0] <= result[0].shape[0] <= TRAIN_SIZE[1])
        self.assertTrue(TRAIN_SIZE[0] <= result[1].shape[0] <= TRAIN_SIZE[1])
        self.assertTrue(TRAIN_SIZE[0] <= result[2].shape[0] <= TRAIN_SIZE[1])
        values, counts = np.unique(np.hstack(result), return_counts=True)
        np.testing.assert_array_equal(values, np.arange(TARGET.shape[0]))
        self.assertTrue(np.all(counts == 2))  # every observation is included in two sets of training indices

    def test_make_train_folds_2(self):
        cv_groups = fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=True)
        result = fit._make_train_folds(
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

    def test_make_train_folds_3(self):
        cv_groups = fit._process_cv_groups(cv_groups=EXTENDED_CV_GROUPS, target=EXTENDED_TARGET, classify=False)
        result = fit._make_train_folds(
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

    def test_make_train_folds_4(self):
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
            fit._make_train_folds(
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
        train_folds = fit._make_train_folds(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )
        result = fit._make_transforms(features=FEATURES, train_folds=train_folds, transform=None)
        self.assertEqual(len(result), len(train_folds))
        self.assertEqual(result[0], None)

    def test_make_transforms_2(self):
        train_folds = fit._make_train_folds(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )
        transform = StandardScaler()
        result = fit._make_transforms(features=FEATURES, train_folds=train_folds, transform=transform)
        self.assertEqual(len(result), len(train_folds))
        self.assertTrue(isinstance(result[0], StandardScaler))
        self.assertFalse(result[0] is transform)  # ensure original input is copied
        np.testing.assert_allclose(result[0].mean_, np.mean(FEATURES[train_folds[0], :], axis=0), atol=1e-5)
        np.testing.assert_allclose(result[0].var_, np.var(FEATURES[train_folds[0], :], axis=0), atol=1e-5)

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

    # functions _execute_plan() and _fit_step() are already covered by tests for fit.select_hyperparameters()

    @staticmethod
    def test_prepare_features_1():
        train_ix = fit._make_train_folds(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )[0]
        result = fit._prepare_features(features=FEATURES, sample_ix=train_ix, transform=None)
        np.testing.assert_array_equal(result, FEATURES[train_ix, :])

    @staticmethod
    def test_prepare_features_2():
        transform = StandardScaler().fit(FEATURES)
        train_ix = fit._make_train_folds(
            features=FEATURES, target=TARGET, cv_groups=None, splitter=StratifiedKFold(n_splits=3)
        )[0]
        result = fit._prepare_features(features=FEATURES, sample_ix=train_ix, transform=transform)
        np.testing.assert_allclose(result, transform.transform(FEATURES[train_ix, :]), atol=1e-5)

    # function _fit_with_subsampling() already covered by tests for fit.select_hyperparameters()

    @staticmethod
    def test_invert_index_1():
        complement = fit._invert_index(index=np.array([1, 3, 5]), max_value=6)
        np.testing.assert_array_equal(complement, np.array([0, 2, 4]))

    @staticmethod
    def test_collect_cv_results_1():
        step_results = [
            (0, 0, np.array([0.0, 1.0, 2.0, 3.0])),
            (0, 1, np.array([8.0, 9.0, 10.0, 11.0])),
            (0, 2, np.array([16.0, 17.0, 18.0, 19.0])),
            (1, 0, np.array([4.0, 5.0, 6.0, 7.0])),
            (1, 1, np.array([12.0, 13.0, 14.0, 15.0])),
            (1, 2, np.array([20.0, 21.0, 22.0, 23.0]))
        ]
        result = fit._collect_cv_results(step_results=step_results, num_folds=3, num_trials=2)
        np.testing.assert_array_equal(result, np.vstack([
            np.arange(0, 8, dtype=float), np.arange(8, 16, dtype=float), np.arange(16, 24, dtype=float)
        ]).transpose())

    def test_evaluate_search_1(self):
        settings = {"lambda_v_grid": np.array([1e-3, 1e-2]), "max_batches": 2}
        step_results = [
            (0, 0, np.array([-1.0, -0.37, -0.33])),
            (0, 1, np.array([-1.0, -0.47, -0.43])),
            (0, 2, np.array([-1.0, -0.57, -0.53])),
            (1, 0, np.array([-1.0, -0.40, -0.37])),
            (1, 1, np.array([-1.0, -0.50, -0.47])),
            (1, 2, np.array([-1.0, -0.60, -0.57]))
        ]  # standard deviation normalized by N - 1 is 0.1 for each experiment
        cv_results = fit._collect_cv_results(step_results=step_results, num_folds=3, num_trials=2)
        result = fit._evaluate_search(cv_results=cv_results, settings=settings)
        self.assertEqual(len(result), 4)
        self.assertEqual(result["cv_results"].shape, (6, 3))
        np.testing.assert_allclose(
            result["cv_results"]["lambda_v"].values, np.hstack([1e-3 * np.ones(3), 1e-2 * np.ones(3)])
        )
        np.testing.assert_allclose(result["cv_results"]["num_batches"].values, np.hstack([np.arange(3), np.arange(3)]))
        np.testing.assert_allclose(
            result["cv_results"]["mean_score"].values, np.array([-1.0, -0.47, -0.43, -1.0, -0.50, -0.47])
        )
        self.assertAlmostEqual(result["threshold"], -0.53)
        self.assertEqual(result["best_ix"], 2)
        self.assertEqual(result["selected_ix"], 4)

    # function _make_final_model() already covered by tests for fit.select_hyperparameters()
