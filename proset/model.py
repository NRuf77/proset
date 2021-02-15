"""Implementation of prototype set models with sklearn compatible interface.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks progress on model fitting at log
level INFO. The invoking application needs to manage log output.
"""

from abc import ABCMeta, abstractmethod
import logging

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from proset.set_manager import ClassifierSetManager
from proset.objective import ClassifierObjective


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LOG_CAPTION = "  ".join(["{:>10s}"] * 6 + ["{:s}"]).format(
   "Iterations", "Calls", "Objective", "Gradient", "Features", "Prototypes", "Status"
)
LOG_MESSAGE = "  ".join(["{:10d}", "{:10d}", "{:10.1e}", "{:10.1e}", "{:10d}", "{:10d}", "{:s}"])

LIMITED_M = 10  # parameters controlling L-BFGS-B fit
LIMITED_FACTR = 1e7
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
            lambda_v=1e-5,
            lambda_w=1e-8,
            alpha_v=0.05,
            alpha_w=0.05,
            num_candidates=1000,
            max_fraction=0.5,
            random_state=None,
            warm_start=False
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
        :param random_state: instance of np.random.RandomState, integer, or None; if a random state is passed, that
            state will be used for randomization; if an integer or None is passed, a new random state is generated
            internally using the argument as seed
        :param warm_start: boolean; whether to create a new model if calling fit() or to add batches to an existing
            model
        """
        self.n_iter = n_iter
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.num_candidates = num_candidates
        self.max_fraction = max_fraction
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(self, X, y, sample_weight=None):
        """Fit proset model to data.

        :param X: 2D numpy float array; feature matrix; sparse matrices or infinite/missing values not supported
        :param y: list-like object; target for supervised learning
        :param sample_weight: 1D numpy array of positive floats or None; sample weights used for likelihood calculation;
            pass None to use unit weights
        :return: no return value; model updated in place
        """
        logger.info("Fit proset model with {} batches and penalties lambda_v = {:0.2e}, lambda_w = {:0.2e}".format(
            self.n_iter, self.lambda_v, self.lambda_w
        ))
        self._check_hyperparameters()
        X, y = self._validate_X_y(X, y, reset=not self.warm_start)
        MySetManager, MyObjective = self._get_compute_classes()
        if not self.warm_start or not hasattr(self, "set_manager"):
            self.set_manager_ = MySetManager(target=y)
        for i in range(self.n_iter):
            objective = MyObjective(
                features=X,
                target=y,
                weights=check_array(sample_weight, ensure_2d=False) if sample_weight is not None else None,
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
                factr=LIMITED_FACTR,
                pgtol=LIMITED_PGTOL,
                maxfun=LIMITED_MAXFUN,
                maxiter=LIMITED_MAXITER,
                maxls=LIMITED_MAXLS
            )
            batch_info = objective.get_batch_info(solution[0])  # solution[0] is the parameter vector
            self.set_manager_.add_batch(batch_info)
            if logger.isEnabledFor(logging.INFO):
                logger.info("Batch {} fit results".format(i + 1))
                logger.info(LOG_CAPTION)
                logger.info(LOG_MESSAGE.format(
                    solution[2]["nit"],
                    solution[2]["funcalls"],
                    solution[1],
                    np.max(np.abs(solution[2]["grad"])),
                    len(np.nonzero(batch_info["feature_weights"])[0]),
                    len(np.nonzero(batch_info["prototype_weights"])[0]),
                    self._parse_solver_status(solution[2])
                ))
        logger.info("Model fit complete")
        return self

    def _check_hyperparameters(self):
        """Check that model hyperparameters are valid.

        :return: no return value; raise a ValueError if an issue is found
        """
        if self.n_iter < 0:
            raise ValueError("Parameter n_iter must not be negative.")
        # validation of other parameters is left to the classes or functions relying on them

    # noinspection PyMethodMayBeStatic, PyUnresolvedReferences
    def _validate_X_y(self, X, y, reset):
        """Check or transform input target and features as appropriate for the model.

        :param X: see docstring of fit() for details
        :param y: see docstring of fit() for details
        :param reset: boolean; whether to prepare the model for a new fit or enable warm start
        :return: transformed versions of X and y; may also update the state of the model instance
        """
        check_classification_targets(y)
        X = check_array(X)
        if reset or not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]
            # the n_features_in_ attribute for tabular input is an sklearn convention
        elif self.n_features_in_ != X.shape[1]:
            raise ValueError(
                "Parameter X must have {} columns for a warm start, same as for previous calls to fit().".format(
                    self.n_features_in_
                )
            )
        return X, y

    @staticmethod
    @abstractmethod
    def _get_compute_classes():
        """Provide classes implementing the set manager and objective function for the model.

        :return: subclasses of proset.set_manager.SetManager and proset.objective.Objective
        """
        raise NotImplementedError("Abstract method Model._get_compute_classes() has no default implementation.")

    @staticmethod
    def _parse_solver_status(solver_status):
        """Translate L-BFGS-B solver status into human-readable format.

        :param solver_status: dict; third output argument of scipy.fmin_l_bfgs_b()
        :return: string
        """
        if solver_status["warnflag"] == 0:
            return "converged"
        if solver_status["warnflag"] == 1:
            return "reached limit on iterations or function calls"
        return "not converged ({})".format(solver_status["task"])

    def predict(self, X, n_iter=None):
        """Predict class labels for a feature matrix.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :return: 1D numpy array or list of 1D numpy arrays; if n_iter is integer or None, a single set of predictions is
            returned as an array; if n_iter is an array, a list of predictions is returned with one element for each
            element of the array
        """
        check_is_fitted(self, attributes="set_manager_")
        return self._compute_prediction(features=check_array(X), num_batches=n_iter)

    @abstractmethod
    def _compute_prediction(self, features, num_batches):
        """Compute prediction.

        :param features: see docstring of predict() for details
        :param num_batches: as the n_iter parameter for predict()
        :return: see docstring of predict() for details
        """
        raise NotImplementedError("Abstract method Model._get_prediction() has no default implementation.")

    def score(self, X, y, sample_weight=None, n_iter=None):
        """Use trained model to score sample data.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param y: list-like object; target for supervised learning
        :param sample_weight: 1D numpy array of positive floats or None; sample weights used for likelihood calculation;
            pass None to use unit weights
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :return: float or 1D numpy array of floats; if n_iter is integer or None, a single score is returned as a float
            value; if n_iter is an array, an array of scores of the same length is returned
        """
        check_is_fitted(self, attributes="set_manager_")
        check_classification_targets(y)
        return self._compute_score(
            features=check_array(X),
            target=y,
            weights=check_array(sample_weight, ensure_2d=False) if sample_weight is not None else None,
            num_batches=n_iter
        )

    @abstractmethod
    def _compute_score(self, features, target, weights, num_batches):
        """Compute score.

        :param features: see docstring of score() for details
        :param target: see docstring of score() for details
        :param weights: see docstring of score() for details
        :param num_batches: as the n_iter parameter for score()
        :return: as return value of score()
        """
        raise NotImplementedError("Abstract method Model._compute_score() has no default implementation.")


# noinspection PyPep8Naming, PyAttributeOutsideInit
class ClassifierModel(Model):
    """Prototype set classifier.
    """

    _estimator_type = "classifier"

    def _validate_X_y(self, X, y, reset):
        """Check or transform input target and features as appropriate for the model.

        :param X: see docstring of Model.fit() for details
        :param y: see docstring of Model.fit() for details
        :param reset: boolean; whether to prepare the model for a new fit or enable warm start
        :return: transformed versions of X and y; may also update the state of the model instance
        """
        X, y = Model._validate_X_y(self, X=X, y=y, reset=reset)
        if reset or not hasattr(self, "label_encoder_"):
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
            self.classes_ = self.label_encoder_.classes_
            # storing classes_ in the main estimator is an sklearn convention
        return X, self.label_encoder_.transform(y)

    @staticmethod
    def _get_compute_classes():
        """Provide classes implementing the set manager and objective function for the model.

        :return: subclasses of proset.set_manager.SetManager and proset.objective.Objective
        """
        return ClassifierSetManager, ClassifierObjective

    def _compute_prediction(self, features, num_batches):
        """Compute prediction.

        :param features: see docstring of Model.predict() for details
        :param num_batches: as the n_iter parameter for Model.predict()
        :return: see docstring of Model.predict() for details
        """
        prediction = self.set_manager_.evaluate(features=features, num_batches=num_batches)
        prediction = [self.classes_[np.argmax(p, axis=1)] for p in prediction]
        if isinstance(num_batches, np.ndarray):
            return prediction
        return prediction[0]

    def _compute_score(self, features, target, weights, num_batches):
        """Compute log-likelihood (non-negative so it works with sklearn cross-validation).

        :param features: see docstring of Model.score() for details
        :param target: see docstring of Model.score() for details
        :param weights: see docstring of Model.score() for details
        :param num_batches: as the n_iter parameter for Model.score()
        :return: as return value of Model.score()
        """
        target = self.label_encoder_.transform(target)
        prediction = self.set_manager_.evaluate(features=features, num_batches=num_batches)
        prediction = [np.take_along_axis(p, target[:, None], axis=1) for p in prediction]
        # keep only probability assigned to true class
        if weights is None:
            prediction = [np.mean(np.log(p)) for p in prediction]
        else:
            total_weight = np.sum(weights)
            prediction = [np.inner(np.log(p), weights) / total_weight for p in prediction]
        if isinstance(num_batches, np.ndarray):
            return np.array(prediction)
        return prediction[0]

    def predict_proba(self, X, n_iter=None):
        """Predict class probabilities for a feature matrix.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :return: 1D numpy array or list of 1D numpy arrays; if n_iter is integer or None, a single set of predictions is
            returned as an array; if n_iter is an array, a list of predictions is returned with one element for each
            element of the array
        """
        check_is_fitted(self, attributes="set_manager_")
        # noinspection PyUnresolvedReferences
        prediction = self.set_manager_.evaluate(features=check_array(X), num_batches=n_iter)
        if isinstance(n_iter, np.ndarray):
            return prediction
        return prediction[0]
