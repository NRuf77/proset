"""Implementation of prototype set models with sklearn compatible interface.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks progress on model fitting at log
level INFO. The invoking application needs to manage log output.
"""

from abc import ABCMeta, abstractmethod
import logging

import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state
from statsmodels.distributions.empirical_distribution import ECDF

from proset.objectives.np_classifier_objective import NpClassifierObjective
from proset.objectives.tf_classifier_objective import TfClassifierObjective
from proset.set_manager import ClassifierSetManager
import proset.shared as shared


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
LOG_START = \
    "Fit proset model with {} batches, lambda_v = {:0.2e}, lambda_w = {:0.2e}, alpha_v={:0.2f}, and alpha_w={:0.2f}"
LOG_RESULT_BATCHES = "Batch {} fit results"
LOG_RESULT_CAPTION = "  ".join(["{:>10s}"] * 6 + ["{:s}"]).format(
   "Iterations", "Calls", "Objective", "Gradient", "Features", "Prototypes", "Status"
)
LOG_RESULT_MESSAGE = "  ".join(["{:10d}", "{:10d}", "{:10.1e}", "{:10.1e}", "{:10d}", "{:10d}", "{:s}"])
LOG_DONE = "Model fit complete"

LIMITED_M = 10  # parameters controlling L-BFGS-B fit
LIMITED_PGTOL = 1e-5
LIMITED_MAXFUN = 15000
LIMITED_MAXITER = 15000
LIMITED_MAXLS = 20


# noinspection PyPep8Naming, PyAttributeOutsideInit
class Model(BaseEstimator, metaclass=ABCMeta):
    """Base class for prototype set models.
    """

    def __init__(
            self,
            n_iter=1,
            lambda_v=1e-3,
            lambda_w=1e-8,
            alpha_v=0.95,
            alpha_w=0.95,
            num_candidates=1000,
            max_fraction=0.5,
            solver_factr=1e7,
            use_tensorflow=False,
            random_state=None
    ):
        """Initialize prototype set model with hyperparameters.

        :param n_iter: non-negative integer; number of batches of prototypes to fit
        :param lambda_v: non-negative float; penalty weight for the feature weights
        :param lambda_w: non-negative float; penalty weight for the prototype weights
        :param alpha_v: float in [0.0, 1.0]; fraction of lambda_v assigned as l2 penalty weight to feature weights; the
            remainder is assigned as l1 penalty weight
        :param alpha_w: float in [0.0, 1.0]; fraction of lambda_w assigned as l2 penalty weight to prototype weights;
            the remainder is assigned as l1 penalty weight
        :param num_candidates: positive integer; number of candidates for prototypes to try for each batch
        :param max_fraction: float in (0.0, 1.0); maximum fraction of candidates to draw from one group of candidates;
            candidates are grouped by class and whether the current model classifies them correctly or not
        :param solver_factr: positive float; this is parameter factr of scipy.optimize.fmin_l_bfgs_b() which controls
            when the algorithm stops because the change in objective function is too small; the documentation recommends
            1e12 for low accuracy, 1e7 for moderate accuracy and 1e1 for extremely high accuracy
        :param use_tensorflow: boolean; whether to use tensorflow for model fitting or rely on numpy only; scoring
            always uses numpy only
        :param random_state: instance of np.random.RandomState, integer, or None; if a random state is passed, that
            state will be used for randomization; if an integer or None is passed, a new random state is generated using
            the argument as seed for every call to fit()
        """
        self.n_iter = n_iter
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.num_candidates = num_candidates
        self.max_fraction = max_fraction
        self.random_state = random_state
        self.use_tensorflow = use_tensorflow
        self.solver_factr = solver_factr

    def fit(self, X, y, sample_weight=None, warm_start=False):
        """Fit proset model to data.

        :param X: 2D numpy float array; feature matrix; sparse matrices or infinite/missing values not supported
        :param y: list-like object; target for supervised learning
        :param sample_weight: 1D numpy array of non-negative floats or None; sample weights used for likelihood
            calculation; pass None to use unit weights
        :param warm_start: boolean; whether to create a new model or to add batches to an existing model
        :return: no return value; model updated in place
        """
        self._check_hyperparameters()
        X, y, sample_weight = self._validate_arrays(X=X, y=y, sample_weight=sample_weight, reset=not warm_start)
        logger.info(LOG_START.format(self.n_iter, self.lambda_v, self.lambda_w, self.alpha_v, self.alpha_w))
        MySetManager, MyObjective = self._get_compute_classes(self.use_tensorflow)  # pylint: disable=invalid-name
        if not warm_start or not hasattr(self, "set_manager_"):
            self.set_manager_ = MySetManager(  # pylint: disable=attribute-defined-outside-init
                target=y, weights=sample_weight
            )
        for i in range(self.n_iter):
            objective = MyObjective(
                features=X,
                target=y,
                weights=sample_weight,
                num_candidates=self.num_candidates,
                max_fraction=self.max_fraction,
                set_manager=self.set_manager_,
                lambda_v=self.lambda_v,
                lambda_w=self.lambda_w,
                alpha_v=self.alpha_v,
                alpha_w=self.alpha_w,
                random_state=check_random_state(self.random_state)
            )
            starting_point, bounds = objective.get_starting_point_and_bounds()
            solution = fmin_l_bfgs_b(
                func=objective.evaluate,
                x0=starting_point,
                bounds=bounds,
                m=LIMITED_M,
                factr=self.solver_factr,
                pgtol=LIMITED_PGTOL,
                maxfun=LIMITED_MAXFUN,
                maxiter=LIMITED_MAXITER,
                maxls=LIMITED_MAXLS
            )
            batch_info = objective.get_batch_info(solution[0])  # solution[0] is the parameter vector
            self.set_manager_.add_batch(batch_info)
            if logger.isEnabledFor(logging.INFO):  # pragma: no cover
                logger.info(LOG_RESULT_BATCHES.format(i + 1))
                logger.info(LOG_RESULT_CAPTION)
                logger.info(LOG_RESULT_MESSAGE.format(
                    solution[2]["nit"],
                    solution[2]["funcalls"],
                    solution[1],
                    np.max(np.abs(solution[2]["grad"])),
                    len(np.nonzero(batch_info["feature_weights"])[0]),
                    len(np.nonzero(batch_info["prototype_weights"])[0]),
                    self._parse_solver_status(solution[2])
                ))
        logger.info(LOG_DONE)
        return self

    def _check_hyperparameters(self):
        """Check that model hyperparameters are valid.

        :return: no return value; raises a ValueError if an issue is found
        """
        if not np.issubdtype(type(self.n_iter), np.integer):
            raise TypeError("Parameter n_iter must be integer.")
        if self.n_iter < 0:
            raise ValueError("Parameter n_iter must not be negative.")
        if self.solver_factr <= 0.0:
            raise ValueError("Parameter solver_factr must be positive.")
        # validation of other parameters is left to the classes or functions relying on them

    # noinspection PyMethodMayBeStatic, PyUnresolvedReferences
    def _validate_arrays(self, X, y, sample_weight, reset):
        """Check or transform input target, features, and sample weights as appropriate for the model.

        :param X: see docstring of fit() for details
        :param y: see docstring of fit() for details
        :param sample_weight: see docstring of fit() for details
        :param reset: boolean; whether to prepare the model for a new fit or enable warm start
        :return: transformed versions of X and y; may also update the state of the model instance
        """
        X, y = check_X_y(X=X, y=y)
        if reset or not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]  # pylint: disable=attribute-defined-outside-init
            # the n_features_in_ attribute for tabular input is an sklearn convention
        elif self.n_features_in_ != X.shape[1]:
            raise ValueError("Parameter X must have {} columns.".format(self.n_features_in_))
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, **shared.FLOAT_TYPE)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("Parameter sample_weight must have one element per row of X if not None.")
        return check_array(X, **shared.FLOAT_TYPE), self._validate_y(y, reset), sample_weight

    @abstractmethod
    def _validate_y(self, y, reset):  # pragma: no cover
        """Perform checks on estimator target that depend on estimator type.

        :param y: list-like object; target for supervised learning
        :param reset: boolean; whether to prepare the model for a new fit or enable warm start
        :return: numpy array representing y
        """
        raise NotImplementedError("Abstract method Model._validate_y() has no default implementation.")

    @staticmethod
    @abstractmethod
    def _get_compute_classes(use_tensorflow):  # pragma: no cover
        """Provide classes implementing the set manager and objective function for the model.

        :param use_tensorflow: see docstring of __init__() for details
        :return: subclasses of proset.set_manager.SetManager and proset.objective.Objective
        """
        raise NotImplementedError("Abstract method Model._get_compute_classes() has no default implementation.")

    @staticmethod
    def _parse_solver_status(solver_status):
        """Translate L-BFGS-B solver status into human-readable format.

        :param solver_status: dict; third output argument of scipy.fmin_l_bfgs_b()
        :return: string; solver exit status
        """
        if solver_status["warnflag"] == 0:
            return "converged"
        if solver_status["warnflag"] == 1:
            return "reached limit on iterations or function calls"
        return "not converged ({})".format(solver_status["task"])

    def predict(self, X, n_iter=None, compute_familiarity=False):
        """Predict class labels for a feature matrix.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :param compute_familiarity: boolean; whether to compute the familiarity for each sample
        :return: 1D numpy array or list of 1D numpy arrays; if n_iter is integer or None, a single set of predictions is
            returned as an array; if n_iter is an array, a list of predictions is returned with one element for each
            element of the array; if compute_familiarity is True, also returns a 1D numpy float array or list of float
            arrays containing the familiarity of each sample
        """
        check_is_fitted(self, attributes="set_manager_")
        return self._compute_prediction(
            X=check_array(X, **shared.FLOAT_TYPE),
            n_iter=n_iter,
            compute_familiarity=compute_familiarity
        )

    @abstractmethod
    def _compute_prediction(self, X, n_iter, compute_familiarity):  # pragma: no cover
        """Compute prediction.

        :param X: 2D numpy array of type specified by shared.FLOAT_TYPE; feature matrix
        :param n_iter: see docstring of predict() for details
        :param compute_familiarity: see docstring of predict() for details
        :return: as return value of Model.predict()
        """
        raise NotImplementedError("Abstract method Model._get_prediction() has no default implementation.")

    def score(self, X, y, sample_weight=None, n_iter=None):
        """Use trained model to score sample data.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param y: list-like object; target for supervised learning
        :param sample_weight: 1D numpy array of non-negative floats or None; sample weights used for likelihood
            calculation; pass None to use unit weights
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :return: float or 1D numpy array of floats; if n_iter is integer or None, a single score is returned as a float
            value; if n_iter is an array, an array of scores of the same length is returned
        """
        check_is_fitted(self, attributes="set_manager_")
        X, y, sample_weight = self._validate_arrays(X=X, y=y, sample_weight=sample_weight, reset=False)
        return self._compute_score(X=X, y=y, sample_weight=sample_weight, n_iter=n_iter)

    @abstractmethod
    def _compute_score(self, X, y, sample_weight, n_iter):  # pragma: no cover
        """Compute score.

        :param X: 2D numpy array of type specified by shared.FLOAT_TYPE; feature matrix
        :param y: numpy array; target for supervised learning
        :param sample_weight: see docstring of score() for details
        :param n_iter: see docstring of score() for details
        :return: as return value of score()
        """
        raise NotImplementedError("Abstract method Model._compute_score() has no default implementation.")

    def export(
            self,
            n_iter=None,
            train_names=None,
            include_features=True,
            feature_names=None,
            scale=None,
            offset=None
    ):
        """Export information on prototypes and parameters from trained model.

        :param n_iter: non-negative integer, or None; number of batches to use for evaluation; pass None for all
            batches
        :param train_names: list of strings, list of lists of strings, or None; names for the original training samples
            in order; these are associated with the prototypes in the report; if the model was trained using the warm
            start option, provide a list of lists with sample names per batch up to n_iter or the total number of
            batches; pass None to use default names 'sample 0', 'sample 1', etc.
        :param include_features: boolean; whether to include information on relevant features
        :param feature_names: list of strings or None; if not None, must have one element per column of features;
            feature names to be used as column headers; pass None to use default names X0, X1, etc.; only used if
            include_features is True
        :param scale: 1D numpy array of positive floats or None; if not None, must have one element per column of
            features; use this to scale features back to their original values for the report; pass None for no scaling;
            only used if include_features is True
        :param offset: 1D numpy array of floats or None; if not None, must have one element per column of features; use
            this to shift features back to their original values for the report; pass None for no offset; only used if
            include_features is True
        :return: pandas data frame with the following columns; columns containing the feature name are repeated once for
            each active feature; active features are ordered by decreasing weight over batches as per
            set_manager.SetManager.get_feature_weights():
            - batch: non-negative float; integer batch index for prototypes, np.Nan for properties of the baseline
              distribution
            - sample: non-negative float; integer sample index for prototypes, np.Nan for properties of the baseline
              distribution
            - sample name: string; sample name
            - target: varies; target for supervised learning
            - prototype weight: positive float; prototype weight
            - <feature> weight: non-negative float; feature weight for the associated batch, np.NaN means the feature
              plays no role for the batch; only included of include_features is True
            - <feature> value: float; feature value as used by the model; set to np.NaN if the feature weight is np.NaN;
              only included of include_features is True
            - <feature> original: float; original feature value; set to np.NaN if the feature weight is np.Nan; this
              column is not generated if both scale and offset are None; only included of include_features is True
        """
        check_is_fitted(self, attributes="set_manager_")
        feature_columns, include_original, scale, offset = self._check_report_input(
            n_iter=n_iter if n_iter is not None else self.set_manager_.num_batches,
            train_names=train_names,
            feature_names=feature_names,
            num_features=self.n_features_in_,
            scale=scale,
            offset=offset,
            sample_name=None
        )[:4]
        batches = self.set_manager_.get_batches(features=None, num_batches=n_iter)
        report = self._make_prototype_report(batches=batches, train_names=train_names, compute_impact=False)
        if include_features:
            report = pd.concat([report, self._make_feature_report(
                batches=batches,
                feature_columns=feature_columns,
                include_original=include_original,
                scale=scale,
                offset=offset,
                active_features=self.set_manager_.get_feature_weights(num_batches=n_iter)["feature_index"],
                include_similarities=False
            )], axis=1)
        report = report.sort_values(["batch", "prototype weight"], ascending=[True, False])
        report = pd.concat([self._make_baseline_for_export(), report])
        report.reset_index(inplace=True, drop=True)
        return report

    @staticmethod
    def _check_report_input(n_iter, train_names, feature_names, num_features, scale, offset, sample_name):
        """Check input for export() and explain() for consistency and apply defaults.

        :param n_iter: non-negative integer; number of batches to evaluate
        :param train_names: see docstring of export() for details
        :param feature_names: see docstring of export() for details
        :param num_features: positive integer; number of features
        :param scale: see docstring of export() for details
        :param offset: see docstring of export() for details
        :param sample_name: string or None; name used for reference sample
        :return: five return arguments:
            - list of lists of strings; each list contains column names associated with one feature in the report
            - boolean; whether original values need to be included in the report
            - 1D numpy array of type specified by shared.FLOAT_TYPE; scale as input or vector of ones if input is None
            - 1D numpy array of type specified by shared.FLOAT_TYPE; offset as input or vector of zeros if input is None
            - string; sample name as input or default
            Raises an error if a check fails.
        """
        if train_names is not None and n_iter > 0:
            if len(train_names) == 0:
                raise ValueError("Parameter train_names must not be empty, pass None to use default sample names.")
            if isinstance(train_names[0], list) and len(train_names) != n_iter:
                raise ValueError(" ".join([
                    "Parameter train_names must have as many elements as the number of batches to be evaluated",
                    "if passing a list."
                ]))
            # there is no detailed check whether the list(s) of names have the correct length, as the number of samples
            # used to create each batch is not available after fitting; if the lists are shorter than the sample indices
            # for a given batch, report creation fails with an index error later
        feature_names = shared.check_feature_names(
            num_features=num_features,
            feature_names=feature_names,
            active_features=None
        )
        feature_columns = [[
            "{} weight".format(feature_name),
            "{} value".format(feature_name),
            "{} original".format(feature_name),
            "{} similarity".format(feature_name)
        ] for feature_name in feature_names]
        include_original = scale is not None or offset is not None
        scale, offset = shared.check_scale_offset(num_features=num_features, scale=scale, offset=offset)
        if sample_name is None:
            sample_name = "new sample"
        return feature_columns, include_original, scale, offset, sample_name

    @classmethod
    def _make_prototype_report(cls, batches, train_names, compute_impact):
        """Format prototype information for report.

        :param batches: list as generated by set_manager.SetManager.get_batches()
        :param train_names: see docstring of export() for details
        :param compute_impact: boolean; whether to compute the similarity and impact for each prototype relative to a
            reference sample; if True, the information for each non-empty batch needs to contain the key 'similarities'
        :return: pandas data frame with the following columns:
            - batch: positive integer; batch index
            - sample: non-negative integer; sample index for prototypes
            - sample name: string; sample name
            - target: varies; target for supervised learning
            - prototype weight: positive float; prototype weight
            - similarity: float in (0.0, 1.0]; similarity between prototype and reference sample; only included if
              compute_impact is True
            - impact: positive float; impact of prototype on reference sample; only included if compute_impact is True
        """
        parts = [
            cls._format_batch(batch=batch, batch_index=i, train_names=train_names)
            for i, batch in enumerate(batches) if batch is not None
        ]
        if len(parts) > 0:
            report = pd.concat(parts, axis=0)
            report.reset_index(inplace=True, drop=True)
            return report
        columns = ["batch", "sample", "sample name", "target", "prototype weight"]
        if compute_impact:
            columns.extend(["similarity", "impact"])
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _format_batch(batch, batch_index, train_names):
        """Format information for a single batch of prototypes to include in the report.

        :param batch: one element from the output list generated by set_manager.SetManager.get_batches(); must not be
            None
        :param batch_index: non-negative integer; batch index
        :param train_names: see docstring of export() for details
        :return: as return value of _make_prototype_report(); the function determines whether impact needs to be
            computed by checking whether the batch definitions contain the key "similarities"
        """
        if train_names is not None and isinstance(train_names[0], list):
            train_names = train_names[batch_index]
        formatted = {
            "batch": batch_index + 1,
            "sample": batch["sample_index"],
            "sample name": [
                train_names[j] if train_names is not None else "sample {}".format(j) for j in batch["sample_index"]
            ],
            "target": batch["target"],
            "prototype weight": batch["prototype_weights"]
        }
        columns = ["batch", "sample", "sample name", "target", "prototype weight"]
        if "similarities" in batch.keys():
            formatted["similarity"] = np.exp(np.sum(np.log(batch["similarities"] + shared.LOG_OFFSET), axis=1))
            # use sum of logarithms instead of product for numerical stability
            formatted["impact"] = formatted["similarity"] * formatted["prototype weight"]
            columns.extend(["similarity", "impact"])
        return pd.DataFrame(formatted, columns=columns)

    @classmethod
    def _make_feature_report(
            cls,
            batches,
            feature_columns,
            include_original,
            scale,
            offset,
            active_features,
            include_similarities
    ):
        """Format feature information for report.

        :param batches: list as generated by set_manager.SetManager.get_batches()
        :param feature_columns: as first return value of _check_report_input()
        :param include_original: boolean; whether to include original feature values in the report
        :param scale: as third return value of _check_report_input
        :param offset: as fourth return value of _check_report_input
        :param active_features: 1D numpy array of non-negative integers; indices of active features across all batches
        :param include_similarities: boolean; whether to include per-feature similarities in the report; if True, the
            information for each non-empty batch needs to contain the key 'similarities'
        :return: pandas data frame with the following columns:
            - <feature> weight: non-negative float; feature weight for the associated batch, np.NaN means the feature
              plays no role for the batch
            - <feature> value: float; feature value as used by the model; set to np.NaN if the feature weight is np.NaN
            - <feature> original: float; original feature value; set to np.NaN if the feature weight is np.Nan; this
              column is not generated if both scale and offset are None
            - <feature> similarity: float in (0.0, 1.0]; per-feature similarity between the prototype and reference
              sample; this is only included if include_similarities is True
        """
        if active_features.shape[0] == 0:
            return pd.DataFrame()
        return pd.concat([cls._format_feature(
            batches=batches,
            feature_index=i,
            feature_columns=feature_columns,
            include_original=include_original,
            scale=scale,
            offset=offset,
            include_similarities=include_similarities
        ) for i in active_features], axis=1)

    @staticmethod
    def _format_feature(batches, feature_index, feature_columns, include_original, scale, offset, include_similarities):
        """Format information for a single feature.

        :param batches: list as generated by set_manager.SetManager.get_batches()
        :param feature_index: positive integer; index of feature
        :param feature_columns: list of strings; as return value of _check_report_input()
        :param include_original: boolean; whether to include original feature values in the report
        :param scale: as third return value of _check_report_input
        :param offset: as fourth return value of _check_report_input
        :param include_similarities: boolean; whether to include per-feature similarities in the report
        :return: as one set of columns for the return value of _make_feature_report()
        """
        feature_columns = feature_columns[feature_index]
        scale = scale[feature_index]
        offset = offset[feature_index]
        result = []
        for batch in batches:
            if batch is not None:
                position = np.nonzero(feature_index == batch["active_features"])[0]
                if len(position) == 0:  # feature is not used by current batch
                    nan_column = np.NaN * np.zeros(batch["prototype_weights"].shape[0], **shared.FLOAT_TYPE)
                    new_info = pd.DataFrame({
                        feature_columns[0]: nan_column,
                        feature_columns[1]: nan_column
                    }, columns=feature_columns[:2])
                    if include_original:
                        new_info[feature_columns[2]] = nan_column
                    if include_similarities:
                        new_info[feature_columns[3]] = nan_column
                else:
                    new_info = pd.DataFrame({
                        feature_columns[0]: batch["feature_weights"][position[0]],
                        feature_columns[1]: np.reshape(
                            batch["prototypes"][:, position], batch["prototype_weights"].shape[0]
                        )
                    }, columns=feature_columns[:2])
                    if include_original:
                        new_info[feature_columns[2]] = scale * new_info[feature_columns[1]] + offset
                    if include_similarities:
                        new_info[feature_columns[3]] = batch["similarities"][:, position]
                result.append(new_info)
        if len(result) > 0:
            result = pd.concat(result, axis=0)
            result.reset_index(inplace=True, drop=True)
            return result
        columns = feature_columns[:2]
        if include_original:
            columns.append(feature_columns[2])
        if include_similarities:
            columns.append(feature_columns[3])
        return pd.DataFrame(columns=columns)

    @abstractmethod
    def _make_baseline_for_export(self):  # pragma: no cover
        """Format properties of baseline estimator for export().

        :return: as return value of _make_prototype_report() without columns 'similarity' and 'impact'
        """
        raise NotImplementedError("Abstract method Model._get_baseline_for_export() has no default implementation.")

    def explain(
            self,
            X,
            y=None,
            n_iter=None,
            familiarity=None,
            sample_name=None,
            train_names=None,
            include_features=True,
            feature_names=None,
            scale=None,
            offset=None
    ):
        """Use trained model to explain prediction for a single sample in terms of prototypes.

        :param X: 2D numpy array; feature matrix having a single row; sparse matrices or infinite/missing values not
            supported
        :param y: single value; target value for sample to be explained; pass None if true value is not known
        :param n_iter: non-negative integer, or None; number of batches to use for evaluation; pass None for all batches
        :param familiarity: 1D numpy array of non-negative floats or None; if not None, reference values for familiarity
            used to convert absolute familiarity of the new sample to a quantile
        :param sample_name: string or None; name for the new sample; pass None to use name 'new sample'; if both
            sample_name and train_names are None, no names are included in the explanation
        :param train_names: list of strings, list of lists of strings, or None; names for the original training samples
            in order; these are associated with the prototypes in the report; if the model was trained using the warm
            start option, provide a list of lists with sample names per batch up to n_iter or the total number of
            batches; pass None to use default names 'sample 0', 'sample 1', etc.
        :param include_features: boolean; whether to include information on relevant features
        :param feature_names: list of strings or None; if not None, must have one element per column of features;
            feature names to be used as column headers; pass None to use default names X0, X1, etc.; only used if
            include_features is True
        :param scale: 1D numpy array of positive floats or None; if not None, must have one element per column of
            features; use this to scale features back to their original values for the report; pass None for no scaling;
            only used if include_features is True
        :param offset: 1D numpy array of positive floats or None; if not None, must have one element per column of
            features; use this to shift features back to their original values for the report; pass None for no offset;
            only used if include_features is True
        :return: pandas data frame with the following columns; columns containing the feature name are repeated once for
            each active feature; active features are ordered by decreasing weight over batches as per
            set_manager.SetManager.get_feature_weights():
            - batch: non-negative float; integer batch index for prototypes, np.Nan for properties of the baseline
              distribution
            - sample: non-negative float; integer sample index for prototypes, np.Nan for properties of the baseline
              distribution
            - sample name: string; sample name
            - target: varies; target for supervised learning
            - prototype weight: positive float; prototype weight
            - similarity: float in (0.0, 1.0]; similarity between prototype and reference sample
            - impact: positive float; impact of prototype on probability estimate for reference sample; product of
              prototype weight and similarity
            - <varies>: one or more columns describing the contribution of each prototype to the prediction; depends on
              the type of estimator
            - <feature> weight: non-negative float; feature weight for the associated batch, np.NaN means the feature
              plays no role for the batch; only included of include_features is True
            - <feature> value: float; feature value as used by the model; set to np.NaN if the feature weight is np.NaN;
              only included of include_features is True
            - <feature> original: float; original feature value; set to np.NaN if the feature weight is np.Nan; this
              column is not generated if both scale and offset are None; only included of include_features is True
            - <feature> similarity: float in (0.0, 1.0]; per-feature similarity between the prototype and reference
              sample; only included of include_features is True
        """
        check_is_fitted(self, attributes="set_manager_")
        feature_columns, include_original, scale, offset, sample_name = self._check_report_input(
            n_iter=n_iter if n_iter is not None else self.set_manager_.num_batches,
            train_names=train_names,
            feature_names=feature_names,
            num_features=self.n_features_in_,
            scale=scale,
            offset=offset,
            sample_name=sample_name
        )
        X = check_array(X, **shared.FLOAT_TYPE)
        batches = self.set_manager_.get_batches(features=X, num_batches=n_iter)
        report = self._make_prototype_report(batches=batches, train_names=train_names, compute_impact=X is not None)
        report = pd.concat([report, self._make_contribution_report(report)], axis=1)
        active_features = self.set_manager_.get_feature_weights(num_batches=n_iter)["feature_index"]
        if include_features:
            report = pd.concat([report, self._make_feature_report(
                batches=batches,
                feature_columns=feature_columns,
                include_original=include_original,
                scale=scale,
                offset=offset,
                active_features=active_features,
                include_similarities=X is not None
            )], axis=1)
        report = report.sort_values("impact", ascending=False)
        report = pd.concat([self._make_baseline_for_explain(
            X=X,
            y=y,
            n_iter=n_iter,
            familiarity=familiarity,
            sample_name=sample_name,
            include_features=include_features,
            active_features=active_features,
            feature_columns=feature_columns,
            include_original=include_original,
            scale=scale,
            offset=offset
        ), report])
        report.reset_index(inplace=True, drop=True)
        return report

    @abstractmethod
    def _make_contribution_report(self, prototype_report):  # pragma: no cover
        """Format contribution of prototypes to prediction for report.

        :param prototype_report: as return value of _make_prototype_report()
        :return: pandas data frame; format depends on model type
        """
        raise NotImplementedError("Abstract method Model._get_contribution_report() has no default implementation.")

    @abstractmethod
    def _make_baseline_for_explain(
            self,
            X,
            y,
            n_iter,
            familiarity,
            sample_name,
            include_features,
            active_features,
            feature_columns,
            include_original,
            scale,
            offset
    ):  # pragma: no cover
        """Format properties of baseline estimator for explain().

        :param X: 2D numpy array of type specified by shared.FLOAT_TYPE; feature matrix
        :param y: see docstring of explain() for details
        :param n_iter: see docstring of explain() for details
        :param familiarity: see docstring of explain() for details
        :param sample_name: see docstring of explain() for details
        :param include_features: see docstring of explain() for details
        :param active_features: 1D numpy array of non-negative integers; indices of active features across all batches
        :param feature_columns: as first return value of _check_report_input()
        :param include_original: boolean; whether to include original feature values in the report
        :param scale: see docstring of explain() for details
        :param offset: see docstring of explain() for details
        :return: pandas data frame with the same columns as return value of explain()
        """
        raise NotImplementedError("Abstract method Model._get_baseline_for_explain() has no default implementation.")

    def shrink(self):
        """Reduce internal state representation of a fitted model to active features across all batches.

        :return: no return value; adds property active_features_ with feature indices w.r.t. the original training data
        """
        check_is_fitted(self, attributes="set_manager_")
        self.active_features_ = self.set_manager_.shrink()  # pylint: disable=attribute-defined-outside-init
        self.n_features_in_ = self.active_features_.shape[0]  # pylint: disable=attribute-defined-outside-init


# noinspection PyPep8Naming, PyAttributeOutsideInit
class ClassifierModel(Model):
    """Prototype set classifier.
    """

    _estimator_type = "classifier"

    def _validate_y(self, y, reset):
        """Perform checks on classification target.

        :param y: list-like object; target for classification
        :param reset: see docstring of Model._validate_y() for details
        :return: 1D numpy integer array representing class labels as integers from 0 to K - 1; the model also gains two
            new properties:
            - classes_: 1D numpy array of original class labels
            - label_encoder_: object of type sklearn.preprocessing.LabelEncoder used to convert class labels to integers
        """
        check_classification_targets(y)
        if reset or not hasattr(self, "label_encoder_"):
            self.label_encoder_ = LabelEncoder()  # pylint: disable=attribute-defined-outside-init
            self.label_encoder_.fit(y)
            self.classes_ = self.label_encoder_.classes_  # pylint: disable=attribute-defined-outside-init
            # storing classes_ in the main estimator is an sklearn convention
        return self.label_encoder_.transform(y)

    @staticmethod
    def _get_compute_classes(use_tensorflow):
        """Provide classes implementing the set manager and objective function for the model.

        :param use_tensorflow: see docstring of Model._get_compute_classes()
        :return: subclasses of proset.set_manager.SetManager and proset.objective.Objective suitable for classification
        """
        if use_tensorflow:
            return ClassifierSetManager, TfClassifierObjective
        return ClassifierSetManager, NpClassifierObjective

    def _compute_prediction(self, X, n_iter, compute_familiarity):
        """Compute prediction.

        :param X: see docstring of Model._compute_prediction() for details
        :param n_iter: see docstring of Model._compute_prediction() for details
        :param compute_familiarity: see docstring of Model._compute_prediction() for details
        :return: as return value of Model._compute_prediction()
        """
        prediction = self.set_manager_.evaluate(
            features=X, num_batches=n_iter, compute_familiarity=compute_familiarity
        )
        if compute_familiarity:
            familiarity = prediction[1]
            prediction = prediction[0]
        else:
            familiarity = None
        prediction = [self.classes_[np.argmax(p, axis=1)] for p in prediction]
        if isinstance(n_iter, np.ndarray):
            if compute_familiarity:
                return prediction, familiarity
            return prediction
        if compute_familiarity:
            return prediction[0], familiarity[0]
        return prediction[0]

    def _compute_score(self, X, y, sample_weight, n_iter):
        """Compute log-likelihood (not multiplied by -1 so it works with sklearn cross-validation).

        :param X: see docstring of Model._compute_score() for details
        :param y: 1D numpy integer array; class labels for classification as integers from 0 to K - 1
        :param sample_weight: see docstring of Model._compute_score() for details
        :param n_iter: see docstring of Model._compute_score() for details
        :return: as return value of Model._compute_score()
        """
        prediction = self.set_manager_.evaluate(
            features=X,
            num_batches=n_iter,
            compute_familiarity=False
        )
        prediction = [np.squeeze(np.take_along_axis(p, y[:, None], axis=1)) for p in prediction]
        # keep only probability assigned to true class
        if sample_weight is None:
            prediction = [np.mean(np.log(p + shared.LOG_OFFSET)) for p in prediction]
        else:
            total_weight = np.sum(sample_weight)
            prediction = [np.sum(np.log(p + shared.LOG_OFFSET) * sample_weight) / total_weight for p in prediction]
        if isinstance(n_iter, np.ndarray):
            return np.array(prediction)
        return prediction[0]

    def predict_proba(self, X, n_iter=None, compute_familiarity=False):
        """Predict class probabilities for a feature matrix.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :param compute_familiarity: boolean; whether to compute the familiarity for each sample
        :return: 2D numpy array or list of 2D numpy arrays of type specified by shared.FLOAT_TYPE; each row contains the
            estimated class probabilities for the corresponding row in the feature matrix; if n_iter is integer or None,
            a single set of predictions is returned as an array; if n_iter is an array, a list of predictions is
            returned with one element for each element of the array; if compute_familiarity is True, also returns a 1D
            numpy float array of type specified by shared.FLOAT_TYPE or list of float arrays containing the familiarity
            of each sample
        """
        check_is_fitted(self, attributes="set_manager_")
        # noinspection PyUnresolvedReferences
        prediction = self.set_manager_.evaluate(
            features=check_array(X, **shared.FLOAT_TYPE), num_batches=n_iter, compute_familiarity=compute_familiarity
        )
        if isinstance(n_iter, np.ndarray):
            if compute_familiarity:
                return prediction[0], prediction[1]
            return prediction
        if compute_familiarity:
            return prediction[0][0], prediction[1][0]
        return prediction[0]

    def _make_baseline_for_export(self):
        """Format marginal probabilities for export().

        :return: as return value of Model._make_baseline_for_export()
        """
        return pd.DataFrame({
            "batch": np.NaN,
            "sample": np.NaN,
            "sample name": self._format_class_labels(self.classes_),
            "target": range(self.classes_.shape[0]),
            # target column is numeric, look up class labels in sample name column for marginals
            "prototype weight": self.set_manager_.marginals
        }, columns=["batch", "sample", "sample name", "target", "prototype weight"])

    @staticmethod
    def _format_class_labels(classes):
        """Format class labels for report.

        :param classes: 1D numpy array; class labels
        :return: list of strings
        """
        return ["marginal probability class '{}'".format(label) for label in classes]

    def _make_contribution_report(self, prototype_report):
        """Format contribution of prototypes to prediction for report.

        :param prototype_report: see docstring of Model._make_contribution_report() for details
        :return: pandas data frame with the following fields:
            - dominant set: 0 or 1; indicates whether the prototype belongs to the dominant set
            - p class <class>: positive float; contribution of prototype to the estimated probability for the given
              class
        """
        contributions, dominant_set = self._compute_contributions(
            impact=prototype_report["impact"].to_numpy(copy=True),
            target=prototype_report["target"].to_numpy(copy=True),
            marginals=self.set_manager_.marginals
        )
        report = {"dominant set": dominant_set}
        columns = ["dominant set"]
        for i in range(self.classes_.shape[0]):
            column_name = "p class {}".format(i)
            report[column_name] = contributions[:, i]
            columns.append(column_name)
        return pd.DataFrame(report, columns=columns)

    @staticmethod
    def _compute_contributions(impact, target, marginals):
        """Compute contributions of prototypes to the probability estimate for the reference sample.

        :param impact: 1D numpy array of positive floats; impact of prototypes on the reference sample
        :param target: 1D numpy array of non-negative integers; target class corresponding to each prototype
        :param marginals: 1D numpy array of positive floats; marginal probabilities for each class
        :return: two return values:
            - 2D numpy array of non-negative floats; contribution of each prototype to the probability estimated with
              one row per prototype and one column per class
            - 1D numpy integer array; dominant set indicated by 1, other prototypes by 0
        """
        contributions = np.zeros((impact.shape[0], marginals.shape[0]), **shared.FLOAT_TYPE)
        scale = np.sum(impact) + 1.0
        if scale == 1.0:
            # the sample is so far away from the prototypes that the impact of prototypes is below numerical tolerance
            return contributions, np.zeros(contributions.shape[0], dtype=int)
        impact = impact / scale
        contributions[np.arange(contributions.shape[0]), target] = impact
        sort_ix = np.argsort(impact)[-1::-1]
        rank = rankdata(impact[sort_ix], method="dense")
        rank = np.hstack([0, 1 + np.max(rank) - rank])
        # the marginals have rank 0, all prototypes with the largest impact rank 1, etc.
        extended = np.vstack([marginals / scale, contributions[sort_ix, :]])
        extended = np.cumsum(np.add.reduceat(extended, indices=shared.find_changes(rank), axis=0), axis=0)
        # cumulative contribution to probability across ranks
        remainder = 1.0 - np.sum(extended, axis=1)  # remainder of probability missing up to a given rank
        top_two = np.array([np.sort(extended[i, :])[-1:-3:-1] for i in range(extended.shape[0])])
        # two larges probabilities assigned up to a given rank
        dominant_rank = np.nonzero(top_two[:, 0] - top_two[:, 1] <= remainder)[0]
        if len(dominant_rank) > 0:
            dominant_rank = dominant_rank[-1] + 1
            dominant_set = (rank[1:] <= dominant_rank).astype(int)
            # noinspection PyUnresolvedReferences
            dominant_set = dominant_set[np.argsort(sort_ix)]  # undo sorting
        else:
            dominant_set = np.zeros(contributions.shape[0], dtype=int)
        # noinspection PyUnresolvedReferences
        return contributions, dominant_set

    def _make_baseline_for_explain(
            self,
            X,
            y,
            n_iter,
            familiarity,
            sample_name,
            include_features,
            active_features,
            feature_columns,
            include_original,
            scale,
            offset
    ):
        """Format properties of baseline estimator for explain().

        :param X: see docstring of Model._make_baseline_for_explain() for details
        :param y: see docstring of Model._make_baseline_for_explain() for details
        :param n_iter: see docstring of Model._make_baseline_for_explain() for details
        :param familiarity: see docstring of Model._make_baseline_for_explain() for details
        :param sample_name: see docstring of Model._make_baseline_for_explain() for details
        :param include_features: see docstring of Model._make_baseline_for_explain() for details
        :param active_features: see docstring of Model._make_baseline_for_explain() for details
        :param feature_columns: see docstring of Model._make_baseline_for_explain() for details
        :param include_original: see docstring of Model._make_baseline_for_explain() for details
        :param scale: see docstring of Model._make_baseline_for_explain() for details
        :param offset: see docstring of Model._make_baseline_for_explain() for details
        :return: pandas data frame as return value of Model.explain(); the columns describing the contribution of
            prototypes to the classification result are
            - dominant set: integer; 1 indicates the prototype belongs to the dominant set, 0 that it does not
            - p class <class>: for the new sample, the predicted probability to belong to the given class; for other
              rows, the contribution towards that estimate
        """
        probabilities, sample_familiarity = self.predict_proba(X=X, n_iter=n_iter, compute_familiarity=True)
        prediction = self.label_encoder_.inverse_transform(np.array([np.argmax(probabilities)]))
        sample_name += ", prediction '{}'".format(prediction[0])
        if familiarity is not None:
            sample_name += ", familiarity {:.2f}".format(ECDF(familiarity)(sample_familiarity)[0])
        no_content = [np.NaN] * (self.classes_.shape[0] + 1)
        marginals = [np.NaN] + list(self.set_manager_.marginals)
        ones = [np.NaN] + [1.0] * (self.classes_.shape[0])
        report = {
            "batch": no_content,
            "sample": no_content,
            "sample name": [sample_name] + self._format_class_labels(self.classes_),
            "target": [
                self.label_encoder_.transform(np.array([y]))[0] if y is not None else np.NaN
            ] + list(range(self.classes_.shape[0])),
            # target column is numeric, look up class labels in sample name column for marginals
            "prototype weight": marginals,
            "similarity": ones,
            "impact": marginals,
            "dominant set": ones
        }
        columns = [
            "batch", "sample", "sample name", "target", "prototype weight", "similarity", "impact", "dominant set"
        ]
        for i in range(self.classes_.shape[0]):
            column_name = "p class {}".format(i)
            new_column = np.zeros(self.classes_.shape[0] + 1, **shared.FLOAT_TYPE)
            new_column[0] = probabilities[0, i]
            new_column[i + 1] = self.set_manager_.marginals[i] / (sample_familiarity + 1.0)
            report[column_name] = new_column
            columns.append(column_name)
        if include_features:
            for i in active_features:
                report[feature_columns[i][0]] = no_content  # feature weight
                report[feature_columns[i][1]] = [X[0, i]] + no_content[:-1]  # feature value used by the model
                columns.extend(feature_columns[i][:2])
                if include_original:  # original feature value
                    report[feature_columns[i][2]] = [scale[i] * X[0, i] + offset[i]] + no_content[:-1]
                    columns.append(feature_columns[i][2])
                report[feature_columns[i][3]] = no_content  # per-feature similarity
                columns.append(feature_columns[i][3])
        return pd.DataFrame(report, columns=columns)
