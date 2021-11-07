"""Implementation of objective functions for prototype set models.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details

This submodule creates a logger named like itself that logs to a NullHandler and tracks sampling of candidates at log
level INFO. The invoking application needs to manage log output.
"""

from abc import ABCMeta, abstractmethod
import logging

import numpy as np

import proset.shared as shared


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


LOG_CAPTION = "  ".join(["{:>10s}"] * 3).format("Group", "Samples", "Candidates")
LOG_MESSAGE = "  ".join(["{:>10s}", "{:10d}", "{:10d}"])

START_FEATURE_WEIGHT = 10.0  # this is divided by the number of features
START_PROTOTYPE_WEIGHT = 1.0
SPARSE_THRESHOLD_FEATURES = 0.7
# exploit sparse structure if the fraction of feature weights that are non-zero is at most equal to this value


class Objective(metaclass=ABCMeta):
    """Abstract base class for objective functions.
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

        :param features: 2D numpy float array; features of training sample; sparse matrices or infinite/missing values
            not supported
        :param target: numpy array; target for supervised learning; must have as many elements along the first
            dimension as features has rows
        :param weights: 1D numpy array of positive floats or None; sample weights to be used in the likelihood function;
            pass None to give unit weight to all samples
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
            weights = np.ones(features.shape[0], dtype=float)
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
        self._sample_cache = None
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
        if len(weights.shape) != 1:
            raise ValueError("Parameter weights must be a 1D array.")
        if weights.shape[0] != features.shape[0]:
            raise ValueError("Parameter weights must have as many elements as features has rows.")
        if np.any(weights <= 0.0):
            raise ValueError("Parameter weights must not contain negative values.")
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
            - ref_features: 2D numpy array with as many columns as features and a subset of the rows; features of
              reference samples used to compute the likelihood for optimization
            - ref_features_squared: 2D numpy array; values of ref_features squared
            - ref_target: numpy array; target values for the reference set
            - ref_weights: 1D numpy array; weights for the reference samples
            - ref_unscaled: numpy array; unscaled predictions from set_manager for ref_features
            - ref_scale: 1D numpy array; scaling factors for ref_unscaled
            - cand_features: 2D numpy array with as many columns as features and a subset of the rows; features of
              candidates for new prototypes
            - cand_features_squared: 2D numpy array; values of cand_features squared
            - cand_target: 1D numpy array; target values for the candidate set; the set manager requires prototypes to
              have single feature values so any ambiguities caused by, e.g., censoring are resolved during splitting
            - cand_index: 1D numpy array; index vector indicating the training samples used as candidates
            - shrink_sparse_features: list of strings; all keys referencing arguments that can be shrunk if the feature
              weights are sparse
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

    @staticmethod
    @abstractmethod
    def _assign_groups(target, unscaled, scale, meta):  # pragma: no cover
        """Divide training samples into groups for sampling candidates.

        :param target: see docstring of __init__() for details
        :param unscaled: numpy array; unscaled predictions corresponding to the target values
        :param scale: 1D numpy array; scaling factors for unscaled
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
        :return: no return value; log message generated if log level is INFO or lower
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info("Candidate breakdown by group")
            logger.info(LOG_CAPTION)
            for i in range(num_groups):
                is_group = groups == i
                logger.info(LOG_MESSAGE.format(
                    str(i + 1), np.sum(is_group), np.sum(np.logical_and(is_group, candidates))
                ))
            logger.info(LOG_MESSAGE.format("Total", len(groups), np.sum(candidates)))

    # noinspection PyUnusedLocal
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
        ref_features = features[reference]
        cand_features = features[candidates]
        return {
            "ref_features": ref_features,
            "ref_features_squared": ref_features ** 2.0,
            "ref_target": target[reference],
            "ref_weights": weights[reference],
            "ref_unscaled": unscaled[reference],
            "ref_scale": scale[reference],
            "cand_features": cand_features,
            "cand_features_squared": cand_features ** 2.0,
            "cand_target": target[candidates],
            "cand_index": np.nonzero(candidates)[0],
            "shrink_sparse_features": ["ref_features", "ref_features_squared", "cand_features", "cand_features_squared"]
        }

    def get_starting_point_and_bounds(self):
        """Provide starting parameter vector and bounds for optimization.

        :return: two return arguments:
            - 1D numpy array of non-negative floats; total length is number of features plus number of candidates; the
              convention is to include feature weights before candidate weights
            - tuple of tuples of floats; the outer tuple has the same length as the parameter vector, the inner tuples
              all have length two and define lower and upper bounds
        """
        return (
            np.hstack([
                START_FEATURE_WEIGHT * np.ones(self._meta["num_features"]) / self._meta["num_features"],
                START_PROTOTYPE_WEIGHT * np.ones(self._meta["num_parameters"] - self._meta["num_features"])
            ]),
            tuple([(0.0, np.inf) for _ in range(self._meta["num_parameters"])])
        )

    def evaluate(self, parameter):
        """Compute the negative log-likelihood function with elastic net penalty for a given parameter vector.

        :param parameter: 1D numpy array of non-negative floats; total length is number of features plus number of
            candidates; the convention is to include feature weights before candidate weights
        :return: a float value and a 1D numpy float array; function value and gradient
        """
        parameter = self._check_evaluate_parameter(parameter, self._meta)
        feature_weights = parameter[:self._meta["num_features"]]
        prototype_weights = parameter[self._meta["num_features"]:]
        penalty_value, feature_penalty_gradient, prototype_penalty_gradient = self._evaluate_penalty(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            lambda_v=self._lambda_v,
            lambda_w=self._lambda_w,
            alpha_v=self._alpha_v,
            alpha_w=self._alpha_w
        )
        feature_weights, active_features = self._verify_sparseness(
            parameter=feature_weights,
            sparse_threshold=SPARSE_THRESHOLD_FEATURES
        )
        sample_data, self._sample_cache = self._shrink_sample_data(
            sample_data=self._sample_data,
            sample_cache=self._sample_cache,
            active_features=active_features
        )
        similarity = self._compute_similarity(feature_weights=feature_weights, sample_data=sample_data)
        nll_value, feature_gradient, prototype_gradient = self._evaluate_likelihood(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            sample_data=sample_data,
            similarity=similarity,
            meta=self._meta
        )
        return nll_value + penalty_value, np.hstack([
            self._expand_gradient(
                gradient=feature_gradient,
                active_parameters=active_features,
                num_parameters=self._meta["num_features"]
            ) + feature_penalty_gradient,
            prototype_gradient + prototype_penalty_gradient
        ])

    @staticmethod
    def _check_evaluate_parameter(parameter, meta):
        """Check whether input to evaluate() is consistent.

        :parameter: see docstring of evaluate() for details
        :param meta: dict; must have key 'num_parameters' referencing the total number of parameters
        :return: parameters as input but capped below at exactly 0.0 as the solver does not guarantee this
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
        return np.maximum(0.0, parameter)

    @staticmethod
    def _verify_sparseness(parameter, sparse_threshold):
        """Check whether parameter vector is sparse enough to attempt performance optimization.

        :param parameter: 1D numpy array of non-negative floats; feature or prototype weights
        :param sparse_threshold: float between 0.0 and 1.0; threshold for sparseness below which optimization is
            attempted
        :return: two return values:
            - 1D numpy array of non-negative floats; parameter vector reduced to non-zero elements if sufficiently
              sparse; else, returns input
            - 1D numpy array of integer or None; if the parameter vector has been reduced, these are the indices of the
              non-zero elements in the original vector; else, returns None
        """
        active_values = np.nonzero(parameter)[0]
        if len(active_values) / len(parameter) <= sparse_threshold:
            return parameter[active_values], active_values
        return parameter, None

    # noinspection PyUnusedLocal
    @staticmethod
    def _evaluate_penalty(feature_weights, prototype_weights, lambda_v, lambda_w, alpha_v, alpha_w):
        """Compute elastic net penalty and gradient.

        :param feature_weights: 1D numpy array of non-negative floats; feature weights
        :param prototype_weights: 1D numpy array of non-negative floats; prototype weights
        :param lambda_v: see docstring of __init__() for details
        :param lambda_w: see docstring of __init__() for details
        :param alpha_v: see docstring of __init__() for details
        :param alpha_w: see docstring of __init__() for details
        :return: a float value and two 1D numpy float array; penalty value, penalty gradient for feature weights, and
            penalty gradient for prototype weights
        """
        scaled_feature_weights = alpha_v * feature_weights
        scaled_prototype_weights = alpha_w * prototype_weights
        penalty = lambda_v * np.sum(
            0.5 * feature_weights * scaled_feature_weights + feature_weights - scaled_feature_weights
        ) + lambda_w * np.sum(
            0.5 * prototype_weights * scaled_prototype_weights + prototype_weights - scaled_prototype_weights
        )  # no need to use np.abs() for the l1 penalty as weights are constrained to be non-negative
        feature_gradient = lambda_v * (scaled_feature_weights + 1 - alpha_v)
        prototype_gradient = lambda_w * (scaled_prototype_weights + 1 - alpha_w)
        return penalty, feature_gradient, prototype_gradient

    @staticmethod
    def _shrink_sample_data(sample_data, sample_cache, active_features):
        """Reduce sample data to exploit sparse feature weights.

        :param sample_data: as return value of _split_samples()
        :param sample_cache: dict or None; if not None, must contain the following keys:
            - active_features: 1D numpy integer array; active features used to generate the cached sample data
            - sample_data: dict; same format as sample data but with feature matrices reduced to active features
        :param active_features: 1D numpy integer array or None; if not None, index vector of active features
        :return: version of sample_data to use for evaluating the likelihood function; new value for sample cache
        """
        if active_features is None:
            return sample_data, None  # parameter vector is not sparse enough to exploit
        if sample_cache is not None and np.array_equal(active_features, sample_cache["active_features"]):
            return sample_cache["sample_data"], sample_cache  # parameter vector is sparse and cache is still valid
        sample_data = {
            key: value[:, active_features] if key in sample_data["shrink_sparse_features"] else value
            for key, value in sample_data.items()
        }  # shrink designated elements to match new set of active features
        return sample_data, {"active_features": active_features, "sample_data": sample_data}

    # noinspection PyUnusedLocal
    @staticmethod
    def _compute_similarity(feature_weights, sample_data):
        """Compute the impact of each prototype candidate on the distribution of each reference point.

        :param feature_weights: 1D numpy array of non-negative floats; feature weights
        :param sample_data: as return value of _split_samples()
        :return: 2D numpy array of positive floats with one row per sample and one column per prototype
        """
        scaled_reference = sample_data["ref_features"] * feature_weights
        scaled_prototypes = sample_data["cand_features"] * feature_weights
        return shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=scaled_prototypes,
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=np.sum(scaled_prototypes ** 2.0, axis=1)
        )

    @classmethod
    @abstractmethod
    def _evaluate_likelihood(
            cls,
            feature_weights,
            prototype_weights,
            sample_data,
            similarity,
            meta
    ):  # pragma: no cover
        """Compute negative log-likelihood function and gradient without the penalty term.

        :param feature_weights: 1D numpy array of non-negative floats; feature weights
        :param prototype_weights: 1D numpy array of non-negative floats; prototype weights
        :param sample_data: as return value of _split_samples()
        :param similarity: as return value of objective._compute_similarity()
        :param meta: dict; required content depends on subclass implementation
        :return: a float value and two 1D numpy float array; function value, gradient for feature weights, and gradient
            for prototype weights
        """
        return NotImplementedError("Abstract method Objective._evaluate_likelihood() has no default implementation.")

    @staticmethod
    def _expand_gradient(gradient, active_parameters, num_parameters):
        """Expand gradient for non-zero feature weights to full length.

        :param gradient: 1D numpy float array; gradient for active parameters
        :param active_parameters: 1D numpy int array or None; index vector of feature weights used in computation; None
            means all feature weights have been used
        :param num_parameters: positive integer; total number of parameters
        :return: 1D numpy float array; gradient of feature vector expanded to full length by padding with zeros
        """
        if active_parameters is None:
            return gradient
        full_feature_gradient = np.zeros(num_parameters)
        full_feature_gradient[active_parameters] = gradient
        return full_feature_gradient

    def get_batch_info(self, parameter):
        """Provide information on batch of prototypes that can be passed to an appropriate instance of SetManager.

        :param parameter: 1D numpy array of non-negative floats; total length is number of features plus number of
            candidates; the convention is to include feature weights before candidate weights
        :return: dict; keys and values match the input arguments of SetManager.add_batch(), so the dict can be passed to
            that function via the ** operator
        """
        return {
            "prototypes": self._sample_data["cand_features"],
            "target": self._sample_data["cand_target"],
            "feature_weights": parameter[:self._meta["num_features"]],
            "prototype_weights": parameter[self._meta["num_features"]:],
            "sample_index": self._sample_data["cand_index"]
        }


class ClassifierObjective(Objective):
    """Objective function for the proset classifier.
    """

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

        :param features: see docstring of Objective.__init__() for details
        :param target: see docstring of Objective.__init__() for details
        :param weights: see docstring of Objective.__init__() for details
        :param num_candidates: see docstring of Objective.__init__() for details
        :param max_fraction: see docstring of Objective.__init__() for details
        :param lambda_v: see docstring of Objective.__init__() for details
        :param lambda_w: see docstring of Objective.__init__() for details
        :param alpha_v: see docstring of Objective.__init__() for details
        :param alpha_w: see docstring of Objective.__init__() for details
        :return: dict equal to the return value of the base class method with additional key 'counts' referencing the
            vector of sample counts per class; raises a ValueError if a check fails
        """
        meta = Objective._check_init_values(
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
        counts = shared.check_classifier_target(target)
        threshold = cls._compute_threshold(max_fraction)
        too_few_cases = counts < threshold
        if np.any(too_few_cases):
            raise ValueError("\n".join([
                "For the given value of max_fraction, there have to be at least {} samples per class.".format(
                    threshold
                ),
                "The following classes have fewer cases: {}.".format(", ".join(
                    ["{}".format(x) for x in np.nonzero(too_few_cases)[0]]
                ))
            ]))
        meta["counts"] = counts
        return meta

    @staticmethod
    def _compute_threshold(max_fraction):
        """Compute required minimum of samples per class.

        :param max_fraction: see docstring of Objective.__init__() for details
        :return: positive integer
        """
        half_samples = max(np.ceil(0.5 / max_fraction), np.floor(0.5 / (1.0 - max_fraction)) + 1.0)
        return 2 * int(half_samples) - 1

    @staticmethod
    def _assign_groups(target, unscaled, scale, meta):
        """Divide training samples into groups for sampling candidates.

        :param target: see docstring of Objective.__init__() for details
        :param unscaled: numpy array; unscaled predictions corresponding to the target values
        :param scale: 1D numpy array; scaling factors for unscaled
        :param meta: dict; must have key 'counts' referencing the vector of sample counts per class
        :return: two return values:
            - integer; total number of groups mandated by hyperparameters
            - 1D numpy integer array; group assignment to samples as integer from 0 to the number of groups - 1; note
              that the assignment is not guaranteed to contain all group numbers
        """
        groups = 2 * target  # correctly classified samples are assigned an even group number based on their target
        groups[target != np.argmax(unscaled, axis=1)] += 1  # incorrectly classified samples are assigned odd numbers
        # no need to scale as the scales for classification are just the row-sums
        return 2 * meta["counts"].shape[0], groups

    @staticmethod
    def _finalize_split(candidates, features, target, weights, unscaled, scale, meta):
        """Apply sample split into candidates for prototypes and reference points.

        :param candidates: as return value of Objective._sample_candidates()
        :param features: see docstring of Objective.__init__() for details
        :param target: see docstring of Objective.__init__() for details
        :param weights: see docstring of Objective.__init__() for details
        :param unscaled: numpy array; unscaled predictions corresponding to the target values
        :param scale: 1D numpy array; scaling factors for unscaled
        :param meta: dict; must have key 'counts' referencing the vector of sample counts per class
        :return: dict in the format specified for the return value of Objective._split_samples(); this function reorders
            the candidate information such that the target values are in ascending order; it adds the following field:
            - class_matches: 2D numpy boolean array with one row per reference point and one column per prototype
              candidate; indicates whether the respective reference and candidate point have the same target value
            - cand_changes: 1D numpy integer array; index vector indicating changes in the candidate target, including
              zero for the first element
        """
        sample_data = Objective._finalize_split(
            candidates=candidates,
            features=features,
            target=target,
            weights=weights,
            unscaled=unscaled,
            scale=scale,
            meta=meta
        )
        for i in range(meta["counts"].shape[0]):  # loop over classes
            ix = target == i
            class_weight = np.sum(weights[ix])
            class_weight /= class_weight - np.sum(weights[np.logical_and(ix, candidates)])
            ix = sample_data["ref_target"] == i
            sample_data["ref_weights"][ix] = sample_data["ref_weights"][ix] * class_weight
        order = np.argsort(sample_data["cand_target"])
        sample_data["cand_features"] = sample_data["cand_features"][order]
        sample_data["cand_features_squared"] = sample_data["cand_features_squared"][order]
        sample_data["cand_target"] = sample_data["cand_target"][order]
        sample_data["cand_changes"] = shared.find_changes(sample_data["cand_target"])
        sample_data["cand_index"] = sample_data["cand_index"][order]
        sample_data["class_matches"] = np.zeros(
            (sample_data["ref_features"].shape[0], sample_data["cand_features"].shape[0]), dtype=bool
        )
        for i in range(len(meta["counts"])):  # for each class
            sample_data["class_matches"] += np.outer(sample_data["ref_target"] == i, sample_data["cand_target"] == i)
        return sample_data

    @classmethod
    def _evaluate_likelihood(cls, feature_weights, prototype_weights, sample_data, similarity, meta):
        """Compute negative log-likelihood function and gradient without the penalty term.

        :param feature_weights: 1D numpy array of non-negative floats; feature weights
        :param prototype_weights: 1D numpy array of non-negative floats; prototype weights
        :param sample_data: as return value of Objective._split_samples()
        :param similarity: as return value of Objective._compute_similarity()
        :param meta: dict; required content depends on subclass implementation
        :return: a float value and two 1D numpy float array; function value, gradient for feature weights, and gradient
            for prototype weights
        """
        shared_expressions = cls._compute_shared_expressions(
            similarity=similarity,
            sample_data=sample_data,
            prototype_weights=prototype_weights,
            meta=meta
        )
        return cls._compute_negative_log_likelihood(shared_expressions), \
            cls._compute_partial_feature_weights(
                shared_expressions=shared_expressions, feature_weights=feature_weights
            ), \
            cls._compute_partial_prototype_weights(shared_expressions)

    @staticmethod
    def _compute_shared_expressions(similarity, sample_data, prototype_weights, meta):
        """Compute expressions required to evaluate objective function and gradient.

        :param similarity: as return value of objective._compute_similarity()
        :param sample_data: as return value of Objective._split_samples() after processing by
            Objective._shrink_sample_data()
        :param prototype_weights: 1D numpy array of non-negative floats; prototype weights
        :param meta: dict; must have key 'total_weight' referencing the total sample weight
        :return: dict with the following fields:
            - similarity: as input
            - similarity_matched: element-wise product of similarity and sample_data["class_matches"]
            - impact: similarity multiplied by prototype_weights (element-wise per row); columns limited to active
              prototypes if active_prototypes is not None
            - impact_matched: impact multiplied by sample_data["class_matches"]; columns limited to active prototypes if
              active_prototypes is not None
            - cand_features: sample_data["cand_features"] if active_prototypes is None, else
              sample_data["cand_features_sparse"]
            - cand_features_squared: sample_data["cand_features_squared"] if active_prototypes is None, else
              sample_data["cand_features_squared_sparse"]
            - ref_features: sample_data["ref_features"]
            - ref_features_squared: sample_data["ref_features_squared"]
            - ref_unscaled: 1D numpy array of positive floats; unscaled probability estimates for the true class of each
              reference point
            - ref_scale: 1D numpy array of positive floats; scales corresponding to ref_unscaled
            - ref_weights: sample_data["ref_weights"]
            - total_weight: meta["total_weight"]
        """
        similarity_matched = similarity * sample_data["class_matches"]
        impact = similarity * prototype_weights
        ref_unscaled = sample_data["ref_unscaled"].copy()
        ref_unscaled += np.add.reduceat(impact, indices=sample_data["cand_changes"], axis=1)
        # update unscaled probabilities with impact of latest batch
        ref_scale = np.sum(ref_unscaled, axis=1)
        ref_unscaled = np.squeeze(np.take_along_axis(ref_unscaled, sample_data["ref_target"][:, None], axis=1))
        # keep only the weight corresponding to the actual class of each point
        return {
            "similarity": similarity,
            "similarity_matched": similarity_matched,
            "impact": impact,
            "impact_matched": similarity_matched * prototype_weights,
            "cand_features": sample_data["cand_features"],
            "cand_features_squared": sample_data["cand_features_squared"],
            "ref_features": sample_data["ref_features"],
            "ref_features_squared": sample_data["ref_features_squared"],
            "ref_unscaled": ref_unscaled,
            "ref_scale": ref_scale,
            "ref_weights": sample_data["ref_weights"],
            "total_weight": meta["total_weight"]
        }

    @staticmethod
    def _compute_negative_log_likelihood(shared_expressions):
        """Compute negative log-likelihood.

        :param shared_expressions: as return value of _compute_shared_expressions()
        :return: float; negative log-likelihood
        """
        return -1.0 * np.inner(
            np.log(shared_expressions["ref_unscaled"] / shared_expressions["ref_scale"] + shared.LOG_OFFSET),
            shared_expressions["ref_weights"]
        ) / shared_expressions["total_weight"]

    @classmethod
    def _compute_partial_feature_weights(cls, shared_expressions, feature_weights):
        """Compute the vector of partial derivatives w.r.t. the feature weights.

        :param shared_expressions: as return value of _compute_shared_expressions()
        :param feature_weights: 1D numpy array of non-negative floats; feature weights
        :return: 1D numpy float array; partial derivatives w.r.t. feature weights
        """
        part_1 = np.inner(
            cls._quick_compute_part(
                ref_features=shared_expressions["ref_features"],
                ref_features_squared=shared_expressions["ref_features_squared"],
                cand_features=shared_expressions["cand_features"],
                cand_features_squared=shared_expressions["cand_features_squared"],
                impact=shared_expressions["impact"]
            ),
            shared_expressions["ref_weights"] / shared_expressions["ref_scale"]
        )
        part_2 = np.inner(
            cls._quick_compute_part(
                ref_features=shared_expressions["ref_features"],
                ref_features_squared=shared_expressions["ref_features_squared"],
                cand_features=shared_expressions["cand_features"],
                cand_features_squared=shared_expressions["cand_features_squared"],
                impact=shared_expressions["impact_matched"]
            ),
            shared_expressions["ref_weights"] / shared_expressions["ref_unscaled"]
        )
        return (part_2 - part_1) * feature_weights / shared_expressions["total_weight"]

    @staticmethod
    def _quick_compute_part(ref_features, ref_features_squared, cand_features, cand_features_squared, impact):
        """Compute a term for the gradient w.r.t. feature weights.

        :param ref_features: as field 'ref_features' from return value of _compute_shared_expressions()
        :param ref_features_squared: as field 'ref_features_squared' from return value of _compute_shared_expressions()
        :param cand_features: as field 'cand_features' from return value of _compute_shared_expressions()
        :param cand_features_squared: as field 'cand_features_squared' from return value of
            _compute_shared_expressions()
        :param impact: as field 'impact' or 'impact_matched' from return value of _compute_shared_expressions()
        :return: 1D numpy float array; contribution of squared differences between features for reference and candidate
            points, weighted with impact
        """
        part = ref_features_squared.transpose() * np.sum(impact, axis=1)
        part -= 2.0 * (ref_features * np.inner(impact, cand_features.transpose())).transpose()
        return part + np.inner(impact, cand_features_squared.transpose()).transpose()

    @staticmethod
    def _compute_partial_prototype_weights(shared_expressions):
        """Compute the vector of partial derivatives w.r.t. the prototype weights.

        :param shared_expressions: as return value of _compute_shared_expressions()
        :return: 1D numpy float array; partial derivatives w.r.t. prototype weights
        """
        partial_prototype_weights = \
            shared_expressions["similarity_matched"].transpose() / shared_expressions["ref_unscaled"]
        partial_prototype_weights -= shared_expressions["similarity"].transpose() / shared_expressions["ref_scale"]
        return -1.0 * np.inner(
            partial_prototype_weights, shared_expressions["ref_weights"]
        ) / shared_expressions["total_weight"]
