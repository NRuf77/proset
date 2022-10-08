"""Abstract base class for proset objective functions.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks sampling of candidates at log
level INFO. Log level DEBUG also outputs the objective function value, maximum norm of the gradient, and sparseness
information for each call to evaluate(). The invoking application needs to manage log output.
"""

from abc import ABCMeta, abstractmethod
import logging

import numpy as np
from sklearn.utils.validation import check_array

import proset.shared as shared


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
LOG_GROUP_BREAKDOWN = "Candidate breakdown by group"
LOG_GROUP_CAPTION = "  ".join(["{:>10s}"] * 3).format("Group", "Samples", "Candidates")
LOG_GROUP_MESSAGE = "  ".join(["{:>10s}", "{:10d}", "{:10d}"])
LOG_EVALUATE = "objective = {:3.1e}, grad = {:3.1e}, features = {} / {}, prototypes = {} / {}"

START_FEATURE_WEIGHT = 10.0  # this is divided by the number of features
START_PROTOTYPE_WEIGHT = 1.0


class Objective(metaclass=ABCMeta):
    """Abstract base class for proset objective functions.
    """

    def __init__(
            self,
            features,
            target,
            weights,
            num_candidates,
            max_fraction,
            set_manager,
            lambda_v,
            lambda_w,
            alpha_v,
            alpha_w,
            random_state
    ):
        """Initialize objective function.

        :param features: 2D numpy array of type specified by shared.FLOAT_TYPE; features of training sample; sparse
            matrices or infinite/missing values not supported
        :param target: numpy array; target for supervised learning; must have as many elements along the first dimension
            as features has rows
        :param weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE or None; sample
            weights to be used in the likelihood function; pass None to use unit weights
        :param num_candidates: positive integer; total number of candidates for prototypes to be drawn
        :param max_fraction: float in (0.0 1.0); maximum fraction of candidates to be drawn from one group of samples;
            what constitutes a group depends on the kind of model
        :param set_manager: an instance of a subclass of abstract class proset.set_manager.SetManager
        :param lambda_v: non-negative float; penalty weight for the feature weights
        :param lambda_w: non-negative float; penalty weight for the prototype weights
        :param alpha_v: float in [0.0, 1.0]; fraction of lambda_v assigned to the l2 penalty for feature weights; the
            complementary fraction (1 - alpha_v) of lambda_v is assigned to the l1 penalty
        :param alpha_w: float in [0.0, 1.0]; fraction of lambda_w assigned to the l2 penalty for prototype weights; the
            complementary fraction (1 - alpha_w) of lambda_w is assigned to the l1 penalty
        :param random_state: an instance of np.random.RandomState
        """
        if weights is None:
            weights = np.ones(features.shape[0], **shared.FLOAT_TYPE)
        self._meta = self._check_init_values(
            features=features,
            target=target,
            weights=weights,
            num_candidates=num_candidates,
            max_fraction=max_fraction,
            lambda_v=lambda_v,
            lambda_w=lambda_w,
            alpha_v=alpha_v,
            alpha_w=alpha_w
        )
        # meta tracks properties of the fitting problem that may depend on subclass implementation; this is passed to
        # static methods dealing with computations in case overriding requires additional information
        self._sample_data = self._split_samples(
            features=features,
            target=target,
            weights=weights,
            num_candidates=num_candidates,
            max_fraction=max_fraction,
            set_manager=set_manager,
            random_state=random_state,
            meta=self._meta
        )
        self._meta["num_features"] = self._sample_data["cand_features"].shape[1]
        self._meta["num_parameters"] = self._meta["num_features"] + self._sample_data["cand_features"].shape[0]
        # the parameter vector for optimization has one element per feature and candidate point; feature weights are
        # placed first by convention
        self._meta["total_weight"] = np.sum(weights)
        self._lambda_v = lambda_v
        self._lambda_w = lambda_w
        self._alpha_v = alpha_v
        self._alpha_w = alpha_w

    #  pylint: disable=too-many-branches
    @classmethod
    def _check_init_values(
            cls,
            features,
            target,
            weights,
            num_candidates,
            max_fraction,
            lambda_v,
            lambda_w,
            alpha_v,
            alpha_w
    ):
        """Check whether input to __init__() is consistent.

        :param features: see docstring of __init__() for details
        :param target: see docstring of __init__() for details
        :param weights: see docstring of __init__() for details
        :param num_candidates: see docstring of __init__() for details
        :param max_fraction: see docstring of __init__() for details
        :param lambda_v: see docstring of __init__() for details
        :param lambda_w: see docstring of __init__() for details
        :param alpha_v: see docstring of __init__() for details
        :param alpha_w: see docstring of __init__() for details
        :return: empty dict; default value for the object's meta parameter; raises a ValueError if a check fails
        """
        if len(features.shape) != 2:
            raise ValueError("Parameter features must be a 2D array.")
        if features.shape[0] <= 1:
            raise ValueError("The training data needs to contain more than one sample.")
            # this check is mandated by the sklearn estimator test; the error message must specifically mention a
            # keyword like 'one sample'
        if target.shape[0] != features.shape[0]:
            raise ValueError("Parameter target must have as many elements as features has rows.")
        shared.check_float_array(x=features, name="features")
        if len(weights.shape) != 1:
            raise ValueError("Parameter weights must be a 1D array.")
        if weights.shape[0] != features.shape[0]:
            raise ValueError("Parameter weights must have as many elements as features has rows.")
        if np.any(weights < 0.0):
            raise ValueError("Parameter weights must not contain negative values.")
        shared.check_float_array(x=weights, name="weights")
        if not np.issubdtype(type(num_candidates), np.integer):
            raise TypeError("Parameter num_candidates must be integer.")
        if num_candidates <= 0:
            raise ValueError("Parameter num_candidates must be positive.")
        if max_fraction <= 0.0 or max_fraction >= 1.0:
            raise ValueError("Parameter max_fraction must lie in (0.0, 1.0).")
        if lambda_v < 0.0:
            raise ValueError("Parameter lambda_v must not be negative.")
        if lambda_w < 0.0:
            raise ValueError("Parameter lambda_w must not be negative.")
        if alpha_v < 0.0 or alpha_v > 1.0:
            raise ValueError("Parameter alpha_v must lie in [0.0, 1.0].")
        if alpha_w < 0.0 or alpha_w > 1.0:
            raise ValueError("Parameter alpha_w must lie in [0.0, 1.0].")
        return {}

    @classmethod
    def _split_samples(cls, features, target, weights, num_candidates, max_fraction, set_manager, random_state, meta):
        """Split training samples into candidates for new prototypes and reference points for computing likelihood.

        :param features: see docstring of __init__() for details
        :param target: see docstring of __init__() for details
        :param weights: see docstring of __init__() for details
        :param num_candidates: see docstring of __init__() for details
        :param max_fraction: see docstring of __init__() for details
        :param random_state: see docstring of __init__() for details
        :param meta: dict; required content depends on subclass implementation
        :return: dict with at least the following fields, subclasses may store additional information:
            - ref_features: 2D numpy array of type specified by shared.FLOAT_TYPE with as many columns as features and a
              subset of the rows; features of reference samples used to compute the likelihood for optimization
            - ref_features_squared: 2D numpy array of type specified by shared.FLOAT_TYPE; values of ref_features
              squared
            - ref_target: numpy array; target values for the reference set
            - ref_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; weights for the reference samples
            - ref_unscaled: numpy array of type specified by shared.FLOAT_TYPE; unscaled predictions from set_manager
              for ref_features
            - ref_scale: 1D numpy array of type specified by shared.FLOAT_TYPE; scaling factors for ref_unscaled
            - cand_features: 2D numpy array of type specified by shared.FLOAT_TYPE with as many columns as features and
              a subset of the rows; features of candidates for new prototypes
            - cand_features_squared: 2D numpy array of type specified by shared.FLOAT_TYPE; values of cand_features
              squared
            - cand_target: 1D numpy array; target values for the candidate set; the set manager requires prototypes to
              have single feature values so any ambiguities caused by, e.g., censoring are resolved during splitting
            - cand_index: 1D numpy array; index vector indicating the training samples used as candidates
        """
        unscaled, scale = set_manager.evaluate_unscaled(features=features, num_batches=None)[0]
        num_groups, groups = cls._assign_groups(target=target, unscaled=unscaled, scale=scale, meta=meta)
        candidates = cls._sample_candidates(
            num_groups=num_groups,
            groups=groups,
            num_candidates=num_candidates,
            max_fraction=max_fraction,
            random_state=random_state
        )
        cls._log_group_breakdown(num_groups, groups, candidates)
        return cls._finalize_split(
            candidates=candidates,
            features=features,
            target=target,
            weights=weights,
            unscaled=unscaled,
            scale=scale,
            meta=meta
        )

    # noinspection PyUnusedLocal
    @staticmethod
    @abstractmethod
    def _assign_groups(target, unscaled, scale, meta):  # pragma: no cover
        """Divide training samples into groups for sampling candidates.

        :param target: see docstring of __init__() for details
        :param unscaled: numpy array of type specified by shared.FLOAT_TYPE; unscaled predictions corresponding to the
            target values
        :param scale: 1D numpy array of type specified by shared.FLOAT_TYPE; scaling factors for unscaled
        :param meta: dict; required content depends on subclass implementation
        :return: two return values:
            - integer; total number of groups mandated by hyperparameters
            - 1D numpy integer array; group assignment to samples as integer from 0 to the number of groups - 1; note
              that the assignment is not guaranteed to contain all group numbers
        """
        raise NotImplementedError("Abstract method Objective._assign_groups() has no default implementation.")

    @classmethod
    def _sample_candidates(cls, num_groups, groups, num_candidates, max_fraction, random_state):
        """Distribute samples between candidates for prototypes and reference points.

        :param num_groups: as first return value of _assign_groups()
        :param groups: as second return value of _assign_groups()
        :param num_candidates: see docstring of __init__() for details
        :param max_fraction: see docstring of __init__() for details
        :param random_state: see docstring of __init__() for details
        :return: 1D numpy boolean array; indicator vector for the prototype candidates
        """
        samples_per_group = cls._get_group_samples(
            num_groups=num_groups,
            groups=groups,
            num_candidates=num_candidates,
            max_fraction=max_fraction
        )
        candidates = np.zeros_like(groups, dtype=bool)
        for i in range(num_groups):
            ix = np.nonzero(groups == i)[0]
            if len(ix) > 0:
                candidates[random_state.choice(a=ix, size=samples_per_group[i], replace=False)] = True
        return candidates

    @staticmethod
    def _log_group_breakdown(num_groups, groups, candidates):  # pragma: no cover
        """Log breakdown of samples to groups at log level INFO.

        :param num_groups: as first return value of _assign_groups()
        :param groups: as second return value of _assign_groups()
        :param candidates: as return value of _sample_candidates()
        :return: no return value; log message generated if log level is at least INFO
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info(LOG_GROUP_BREAKDOWN)
            logger.info(LOG_GROUP_CAPTION)
            for i in range(num_groups):
                is_group = groups == i
                logger.info(LOG_GROUP_MESSAGE.format(
                    str(i + 1), np.sum(is_group), np.sum(np.logical_and(is_group, candidates))
                ))
            logger.info(LOG_GROUP_MESSAGE.format("Total", len(groups), np.sum(candidates)))

    @staticmethod
    def _get_group_samples(num_groups, groups, num_candidates, max_fraction):
        """Decide on number of samples per group used as candidates for prototypes.

        :param num_groups: as first return value of _assign_groups()
        :param groups: as second return value of _assign_groups()
        :param num_candidates: see docstring of __init__() for details
        :param max_fraction: see docstring of __init__() for details
        :return: 1D numpy integer array of length num_groups; number of samples to be drawn from each group
        """
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        complete_counts = np.zeros(num_groups, dtype=int)  # add groups with zero count
        complete_counts[unique_groups] = group_counts
        unique_groups = np.argsort(complete_counts)  # reorder groups based on ascending size
        group_counts = complete_counts[unique_groups]
        samples_per_group = np.zeros(num_groups)
        to_assign = num_candidates
        for i in range(num_groups):
            equal_division = to_assign / (num_groups - i)
            limit = group_counts[i] * max_fraction
            if equal_division <= limit:
                # all remaining groups are large enough to divide the remainder equally
                samples_per_group[i:] = equal_division
                break
            samples_per_group[i] = limit
            to_assign -= limit
        original_order = np.zeros_like(samples_per_group)
        original_order[unique_groups] = samples_per_group
        return np.round(original_order).astype(int)

    @staticmethod
    def _finalize_split(
            candidates,
            features,
            target,
            weights,
            unscaled,
            scale,
            meta
    ):  # pylint: disable=unused-argument
        """Apply sample split into candidates for prototypes and reference points.

        :param candidates: as return value of _sample_candidates()
        :param features: see docstring of __init__() for details
        :param target: see docstring of __init__() for details
        :param weights: see docstring of __init__() for details
        :param unscaled: numpy array; unscaled predictions corresponding to the target values
        :param scale: 1D numpy array; scaling factors for unscaled
        :param meta: dict; not used by the default implementation
        :return: dict in the format specified for the return value of _split_samples()
        """
        reference = np.logical_not(candidates)
        ref_features = features[reference].astype(**shared.FLOAT_TYPE)  # ensure 2D arrays have specified order
        cand_features = features[candidates].astype(**shared.FLOAT_TYPE)
        return {
            "ref_features": ref_features,
            "ref_features_squared": ref_features ** 2.0,
            "ref_target": target[reference],  # no need to enforce order on 1D array
            "ref_weights": weights[reference],
            "ref_unscaled": unscaled[reference].astype(**shared.FLOAT_TYPE),
            "ref_scale": scale[reference],
            "cand_features": cand_features,
            "cand_features_squared": cand_features ** 2.0,
            "cand_target": target[candidates],
            "cand_index": np.nonzero(candidates)[0]
        }

    def get_starting_point_and_bounds(self):
        """Provide starting parameter vector and bounds for optimization.

        :return: two return arguments:
            - 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; total length is number of
              features plus number of candidates; the convention is to include feature weights before candidate weights
            - tuple of tuples of floats; the outer tuple has the same length as the parameter vector, the inner tuples
              all have length two and define lower and upper bounds
        """
        return (
            np.hstack([
                START_FEATURE_WEIGHT * np.ones(
                    self._meta["num_features"], **shared.FLOAT_TYPE
                ) / self._meta["num_features"],
                START_PROTOTYPE_WEIGHT * np.ones(
                    self._meta["num_parameters"] - self._meta["num_features"], **shared.FLOAT_TYPE
                )
            ]),
            tuple([(0.0, np.inf) for _ in range(self._meta["num_parameters"])])
        )

    def evaluate(self, parameter):
        """Compute the penalized loss function and gradient for a given parameter vector.

        :param parameter: 1D numpy array with non-negative values; total length is number of features plus number of
            candidates; the convention is to include feature weights before candidate weights
        :return: a float value and a 1D numpy array of type float64; function value and gradient
        """
        parameter = self._check_evaluate_parameter(parameter=parameter, meta=self._meta)
        objective, gradient = self._evaluate_objective(parameter)
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug(LOG_EVALUATE.format(
                objective,
                np.max(np.abs(gradient)),
                np.nonzero(parameter[:self._meta["num_features"]])[0].shape[0],
                self._meta["num_features"],
                np.nonzero(parameter[self._meta["num_features"]:])[0].shape[0],
                parameter.shape[0] - self._meta["num_features"]
            ))
        return objective, gradient.astype(np.float64)

    @staticmethod
    def _check_evaluate_parameter(parameter, meta):
        """Check whether input to evaluate() is consistent.

        :parameter: see docstring of evaluate() for details
        :param meta: dict; must have key 'num_parameters' referencing the total number of parameters
        :return: parameters capped below at exactly 0.0 as the solver does not guarantee this; array type is converted
            to the one specified by shared.FLOAT_TYPE if input is different
        """
        if len(parameter.shape) != 1:
            raise ValueError("Parameter parameter must be a 1D array.")
        if parameter.shape[0] != meta["num_parameters"]:
            raise ValueError(
                " ".join([
                    "Parameter parameter must have as many elements as the number of features and candidates",
                    "(expected {}, found {}).".format(meta["num_parameters"], parameter.shape[0])
                ])
            )
        return np.maximum(0.0, check_array(parameter, ensure_2d=False, **shared.FLOAT_TYPE))

    # noinspection PyUnusedLocal
    @abstractmethod
    def _evaluate_objective(self, parameter):  # pragma: no cover
        """Actually compute the penalized loss function and gradient for a given parameter vector.

        :param parameter: as return value of Objective._check_evaluate_parameter()
        :return: as return values of Objective.evaluate()
        """
        return NotImplementedError("Abstract method Objective._evaluate_objective() has no default implementation.")

    # noinspection PyUnusedLocal
    @abstractmethod
    def get_batch_info(self, parameter):  # pragma: no cover
        """Provide information on batch of prototypes that can be passed to an appropriate instance of SetManager.

        :param parameter: 1D numpy array of non-negative floats; total length is number of features plus number of
            candidates; the convention is to include feature weights before candidate weights
        :return: dict; keys and values match the input arguments of SetManager.add_batch(), so the dict can be passed to
            that function via the ** operator; all float arrays are converted to the type specified in shared.FLOAT_TYPE
        """
        return NotImplementedError("Abstract method Objective.get_batch_info() has no default implementation.")
