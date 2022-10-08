"""Abstract base class for proset objective functions that use tensorflow to evaluate the likelihood.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from abc import ABCMeta, abstractmethod

import numpy as np
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

from proset.objectives.objective import Objective
from proset.shared import FLOAT_TYPE


class TfObjective(Objective, metaclass=ABCMeta):
    """Abstract base class for proset objective functions that use tensorflow to evaluate the likelihood.
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
        if tf is None:  # pragma: no cover
            raise RuntimeError("Class TfObjective missing optional dependency tensorflow.")
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

    @staticmethod
    def _convert_sample_data(sample_data, exclude):
        """Convert processed sample data to tensorflow constants.

        :param sample_data: as return value of objective.Objective._finalize_split(); additional keys are allowed if
            they are numpy arrays or in the exclude list
        :param exclude: list of strings; keys of sample_data for which the value should not be converted
        :return: no return value; sample_data is updated in place
        """
        for key, value in sample_data.items():
            if key not in exclude:
                sample_data[key] = tf.constant(value, dtype=float)

    def _evaluate_objective(self, parameter):
        """Actually compute the penalized loss function and gradient for a given parameter vector.

        :param parameter: see docstring of objective.Objective._evaluate_objective() for details
        :return: as return values of Objective._evaluate_objective()
        """
        feature_weights = tf.Variable(parameter[:self._meta["num_features"]], dtype=float)
        prototype_weights = tf.Variable(parameter[self._meta["num_features"]:], dtype=float)
        objective, gradient_tape = self._make_gradient_tape(
            feature_weights=feature_weights,
            prototype_weights=prototype_weights,
            sample_data=self._sample_data,
            lambda_v=self._lambda_v,
            lambda_w=self._lambda_w,
            alpha_v=self._alpha_v,
            alpha_w=self._alpha_w,
            meta=self._meta
        )
        gradient = gradient_tape.gradient(objective, [feature_weights, prototype_weights])
        return objective.numpy().astype(np.float64), np.hstack([gradient[0].numpy(), gradient[1].numpy()])
        # to be consistent with numpy-only implementation, return float scalar as 64bit value

    @classmethod
    @abstractmethod
    def _make_gradient_tape(
            cls,
            feature_weights,
            prototype_weights,
            sample_data,
            lambda_v,
            lambda_w,
            alpha_v,
            alpha_w,
            meta
    ):  # pragma: no cover
        """Supply objective function tensor and gradient tape for evaluation.

        :param feature_weights: 1D tensorflow variable; feature weights
        :param prototype_weights: 1D tensorflow variable; prototype weights
        :param sample_data: as return value of objective.Objective._finalize_split(), with relevant content converted to
            tensorflow constants and additional content depending on subclass implementation
        :param lambda_v: see docstring of __init__() for details
        :param lambda_w: see docstring of __init__() for details
        :param alpha_v: see docstring of __init__() for details
        :param alpha_w: see docstring of __init__() for details
        :param meta: dict; required content depends on subclass implementation
        :return: objective function tensor and gradient tape
        """
        raise NotImplementedError("Abstract method TfObjective._make_gradient_tape() has no default implementation.")

    @staticmethod
    def _quick_compute_similarity(scaled_reference, scaled_prototypes, ssq_reference, ssq_prototypes):
        """Compute similarity between prototypes and reference points.

        :param scaled_reference: 2D float tensor; features for reference points scaled with feature weights
        :param scaled_prototypes: 2D float tensor; features for prototypes scaled with feature weights; must have as
            many columns as scaled_reference
        :param ssq_reference: 1D float tensor; the row-sums of scaled_reference after squaring the values
        :param ssq_prototypes: 1D float tensor; the row-sums of scaled_prototypes after squaring the values
        :return: 2D tensor of positive floats with one row per sample and one column per prototype
        """
        similarity = -2.0 * tf.matmul(scaled_reference, tf.transpose(scaled_prototypes))
        similarity += ssq_prototypes
        similarity = tf.transpose(tf.transpose(similarity) + ssq_reference)  # broadcast over columns
        similarity = tf.exp(-0.5 * similarity)
        return similarity

    @staticmethod
    def _evaluate_penalty(feature_weights, prototype_weights, lambda_v, lambda_w, alpha_v, alpha_w):
        """Compute elastic net penalty and gradient.

        :param feature_weights: 1D tensor of non-negative floats; feature weights
        :param prototype_weights: 1D tensor of non-negative floats; prototype weights
        :param lambda_v: see docstring of objective.Objective.__init__() for details
        :param lambda_w: see docstring of objective.Objective.__init__() for details
        :param alpha_v: see docstring of objective.Objective.__init__() for details
        :param alpha_w: see docstring of objective.Objective.__init__() for details
        :return: scalar tensor
        """
        scaled_feature_weights = alpha_v * feature_weights
        scaled_prototype_weights = alpha_w * prototype_weights
        return lambda_v * tf.reduce_sum(
            0.5 * feature_weights * scaled_feature_weights + feature_weights - scaled_feature_weights
        ) + lambda_w * tf.reduce_sum(
            0.5 * prototype_weights * scaled_prototype_weights + prototype_weights - scaled_prototype_weights
        )  # no need to use np.abs() for the l1 penalty as weights are constrained to be non-negative

    def get_batch_info(self, parameter):
        """Provide information on batch of prototypes that can be passed to an appropriate instance of SetManager.

        :param parameter: see docstring of objective.Objective.get_batch_info() for details
        :return: as return value of objective.Objective.get_batch_info()
        """
        return {
            "prototypes": self._sample_data["cand_features"].numpy().astype(**FLOAT_TYPE),
            "target": self._sample_data["cand_target"].numpy().astype(int),
            "feature_weights": parameter[:self._meta["num_features"]].astype(**FLOAT_TYPE),
            "prototype_weights": parameter[self._meta["num_features"]:].astype(**FLOAT_TYPE),
            "sample_index": self._sample_data["cand_index"]
        }
