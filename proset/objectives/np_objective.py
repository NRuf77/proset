"""Abstract base class for proset objective functions that use numpy to evaluate the likelihood.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from proset.objectives.objective import Objective
import proset.shared as shared


SPARSE_THRESHOLD_FEATURES = 0.7
# exploit sparse structure if the fraction of feature weights that are non-zero is at most equal to this value


class NpObjective(Objective, metaclass=ABCMeta):
    """Abstract base class for proset objective functions that use numpy to evaluate the likelihood.
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

        :param features: see docstring of objective.Objective.__init__() for details
        :param target: see docstring of objective.Objective.__init__() for details
        :param weights: see docstring of objective.Objective.__init__() for details
        :param num_candidates: see docstring of objective.Objective.__init__() for details
        :param max_fraction: see docstring of objective.Objective.__init__() for details
        :param set_manager: see docstring of objective.Objective.__init__() for details
        :param lambda_v: see docstring of objective.Objective.__init__() for details
        :param lambda_w: see docstring of objective.Objective.__init__() for details
        :param alpha_v: see docstring of objective.Objective.__init__() for details
        :param alpha_w: see docstring of objective.Objective.__init__() for details
        :param random_state: see docstring of objective.Objective.__init__() for details
        """
        Objective.__init__(
            self=self,
            features=features,
            target=target,
            weights=weights,
            num_candidates=num_candidates,
            max_fraction=max_fraction,
            set_manager=set_manager,
            lambda_v=lambda_v,
            lambda_w=lambda_w,
            alpha_v=alpha_v,
            alpha_w=alpha_w,
            random_state=random_state
        )
        self._sample_cache = None

    @staticmethod
    def _finalize_split(
            candidates,
            features,
            target,
            weights,
            unscaled,
            scale,
            meta
    ):
        """Apply sample split into candidates for prototypes and reference points.

        :param candidates: see docstring of objective.Objective._finalize_split() for details
        :param features: see docstring of objective.Objective.__init__() for details
        :param target: see docstring of objective.Objective.__init__() for details
        :param weights: see docstring of objective.Objective.__init__() for details
        :param unscaled: see docstring of objective.Objective._finalize_split() for details
        :param scale: see docstring of objective.Objective._finalize_split() for details
        :param meta: see docstring of objective.Objective._finalize_split() for details
        :return: dict in the format specified for the return value of objective.Objective_split_samples() with one
            additional field:
            - shrink_sparse_features: list of strings; all keys referencing arguments that can be shrunk if the feature
              weights are sparse
        """
        split = Objective._finalize_split(
            candidates=candidates,
            features=features,
            target=target,
            weights=weights,
            unscaled=unscaled,
            scale=scale,
            meta=meta
        )
        split["shrink_sparse_features"] = [
            "ref_features", "ref_features_squared", "cand_features", "cand_features_squared"
        ]
        return split

    def _evaluate_objective(self, parameter):
        """Actually compute the penalized loss function and gradient for a given parameter vector.

        :param parameter: see docstring of objective.Objective._evaluate_objective() for details
        :return: as return values of Objective._evaluate_objective()
        """
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
        feature_weights, active_features = self._verify_sparseness(feature_weights)
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
    def _verify_sparseness(parameter):
        """Check whether parameter vector is sparse enough to attempt performance optimization.

        :param parameter: 1D numpy array of non-negative floats; feature or prototype weights
        :return: two return values:
            - 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; parameter vector reduced
              to non-zero elements if sufficiently sparse; else, returns input
            - 1D numpy array of integer or None; if the parameter vector has been reduced, these are the indices of the
              non-zero elements in the original vector; else, returns None
        """
        active_values = np.nonzero(parameter)[0]
        if len(active_values) / len(parameter) <= SPARSE_THRESHOLD_FEATURES:
            return parameter[active_values], active_values
        return parameter, None

    # noinspection PyUnusedLocal
    @staticmethod
    def _evaluate_penalty(feature_weights, prototype_weights, lambda_v, lambda_w, alpha_v, alpha_w):
        """Compute elastic net penalty and gradient.

        :param feature_weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; feature
            weights
        :param prototype_weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE;
            prototype weights
        :param lambda_v: see docstring of __init__() for details
        :param lambda_w: see docstring of __init__() for details
        :param alpha_v: see docstring of __init__() for details
        :param alpha_w: see docstring of __init__() for details
        :return: a float value and two 1D numpy arrays of type specified by shared.FLOAT_TYPE; penalty value, penalty
            gradient for feature weights, and penalty gradient for prototype weights
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

        :param sample_data: as return value of objective.Objective._split_samples()
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

        :param feature_weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; feature
            weights
        :param sample_data: as return value of objective.Objective._split_samples()
        :return: 2D numpy array with positive values of type specified by shared.FLOAT_TYPE with one row per sample and
            one column per prototype
        """
        scaled_reference = sample_data["ref_features"] * feature_weights
        scaled_prototypes = sample_data["cand_features"] * feature_weights
        # computing the matrix-vector product once and squaring the result is faster with numpy than multiplying the
        # pre-computed matrices of squared features with the squared weights
        return shared.quick_compute_similarity(
            scaled_reference=scaled_reference,
            scaled_prototypes=scaled_prototypes,
            ssq_reference=np.sum(scaled_reference ** 2.0, axis=1),
            ssq_prototypes=np.sum(scaled_prototypes ** 2.0, axis=1)
        )

    # noinspection PyUnusedLocal
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

        :param feature_weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; feature
            weights
        :param prototype_weights: 1D with non-negative values of type specified by shared.FLOAT_TYPE; prototype weights
        :param sample_data: as return value of objective.Objective._split_samples()
        :param similarity: as return value of _compute_similarity()
        :param meta: dict; required content depends on subclass implementation
        :return: a float value and two 1D numpy arrays of type specified by shared.FLOAT_TYPE; function value, gradient
            for feature weights, and gradient for prototype weights
        """
        return NotImplementedError("Abstract method Objective._evaluate_likelihood() has no default implementation.")

    @staticmethod
    def _expand_gradient(gradient, active_parameters, num_parameters):
        """Expand gradient for non-zero feature weights to full length.

        :param gradient: 1D numpy array of type specified by shared.FLOAT_TYPE; gradient for active parameters
        :param active_parameters: 1D numpy int array or None; index vector of feature weights used in computation; None
            means all feature weights have been used
        :param num_parameters: positive integer; total number of parameters
        :return: 1D numpy array of type specified by shared.FLOAT_TYPE; gradient of feature vector expanded to full
            length by padding with zeros
        """
        if active_parameters is None:
            return gradient
        full_feature_gradient = np.zeros(num_parameters, **shared.FLOAT_TYPE)
        full_feature_gradient[active_parameters] = gradient
        return full_feature_gradient

    def get_batch_info(self, parameter):
        """Provide information on batch of prototypes that can be passed to an appropriate instance of SetManager.

        :param parameter: see docstring of objective.Objective.get_batch_info() for details
        :return: as return value of objective.Objective.get_batch_info()
        """
        return {
            "prototypes": self._sample_data["cand_features"],
            "target": self._sample_data["cand_target"],
            "feature_weights": parameter[:self._meta["num_features"]].astype(**shared.FLOAT_TYPE),
            "prototype_weights": parameter[self._meta["num_features"]:].astype(**shared.FLOAT_TYPE),
            "sample_index": self._sample_data["cand_index"]
        }
