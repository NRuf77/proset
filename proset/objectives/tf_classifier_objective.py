"""Class for proset classifier objective function that uses tensorflow to evaluate the likelihood.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

from proset.objectives.tf_objective import TfObjective
import proset.objectives.shared_classifier as shared_classifier
from proset.shared import LOG_OFFSET


class TfClassifierObjective(TfObjective):
    """Objective function for a proset classifier that uses tensorflow to evaluate the likelihood.
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

        :param features: see docstring of objective.Objective.__init__() for details
        :param target: 1D numpy integer array; class labels encodes as integers from 0 to K - 1
        :param weights: see docstring of objective.Objective.__init__() for details
        :param num_candidates: see docstring of objective.Objective.__init__() for details
        :param max_fraction: see docstring of objective.Objective.__init__() for details
        :param lambda_v: see docstring of objective.Objective.__init__() for details
        :param lambda_w: see docstring of objective.Objective.__init__() for details
        :param alpha_v: see docstring of objective.Objective.__init__() for details
        :param alpha_w: see docstring of objective.Objective.__init__() for details
        :return: dict equal to the return value of the base class method with additional key 'num_classes' referencing
            the number of classes; raises a ValueError if a check fails
        """
        meta = TfObjective._check_init_values(  # this already checks whether tensorflow is installed
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
        meta["num_classes"] = shared_classifier.check_classifier_init_values(target=target, max_fraction=max_fraction)
        return meta

    @staticmethod
    def _assign_groups(target, unscaled, scale, meta):
        """Divide training samples into groups for sampling candidates.

        :param target: see docstring of _check_init_values() for details
        :param unscaled: 2D numpy array of type specified by shared.FLOAT_TYPE; unscaled predictions corresponding to
            the target values
        :param scale: see docstring of objective.Objective._assign_groups() for details
        :param meta: dict; must have key 'num_classes' referencing the number of classes
        :return: as return value of objective.Objective._assign_groups()
        """
        return shared_classifier.assign_groups(target=target, unscaled=unscaled, meta=meta)

    @staticmethod
    def _finalize_split(candidates, features, target, weights, unscaled, scale, meta):
        """Apply sample split into candidates for prototypes and reference points.

        :param candidates: see docstring of objective.Objective._finalize_split() for details
        :param features: see docstring of objective.Objective._finalize_split() for details
        :param target: see docstring of _check_init_values() for details
        :param weights: see docstring of objective.Objective._finalize_split() for details
        :param unscaled: see docstring of objective.Objective._finalize_split() for details
        :param scale: see docstring of objective.Objective._finalize_split() for details
        :param meta: dict; must have key 'num_classes' referencing the number of classes
        :return: dict in the format specified for the return value of Objective.objective._split_samples(); this
            function adds the following field:
            - class_matches: 2D numpy boolean array with one row per reference point and one column per prototype
              candidate; indicates whether the respective reference and candidate point have the same target value
        """
        sample_data = TfObjective._finalize_split(
            candidates=candidates,
            features=features,
            target=target,
            weights=weights,
            unscaled=unscaled,
            scale=scale,
            meta=meta
        )
        sample_data["ref_unscaled"] = np.squeeze(
            np.take_along_axis(sample_data["ref_unscaled"], sample_data["ref_target"][:, None], axis=1)
        )  # keep only the weight corresponding to the actual class of each point
        sample_data["ref_weights"] = shared_classifier.adjust_ref_weights(
            sample_data=sample_data, candidates=candidates, target=target, weights=weights, meta=meta
        )
        sample_data["class_matches"] = shared_classifier.find_class_matches(sample_data=sample_data, meta=meta)
        TfObjective._convert_sample_data(sample_data, exclude=["cand_index"])
        return sample_data

    @classmethod
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
    ):
        """Supply objective function tensor and gradient tape for evaluation.

        :param feature_weights: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param prototype_weights: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param sample_data: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param lambda_v: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param lambda_w: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param alpha_v: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param alpha_w: see docstring of tf_objective.TfObjective._make_gradient_tape() for details
        :param meta: dict; must have key 'total_weight' referencing the total sample weight
        :return: as return values of tf_objective.TfObjective._make_gradient_tape()
        """
        with tf.GradientTape() as gradient_tape:
            sq_weights = feature_weights ** 2.0
            # multiplying the pre-computed matrices of squared features with the squared weights is faster with
            # tensorflow than computing the matrix-vector product once and squaring the result
            impact = cls._quick_compute_similarity(
                scaled_reference=sample_data["ref_features"] * feature_weights,
                scaled_prototypes=sample_data["cand_features"] * feature_weights,
                ssq_reference=tf.reduce_sum(sample_data["ref_features_squared"] * sq_weights, axis=1),
                ssq_prototypes=tf.reduce_sum(sample_data["cand_features_squared"] * sq_weights, axis=1)
            ) * prototype_weights
            unscaled = tf.reduce_sum(impact * sample_data["class_matches"], axis=1) + sample_data["ref_unscaled"]
            scale = tf.reduce_sum(impact, axis=1) + sample_data["ref_scale"]
            objective = -tf.reduce_sum(tf.math.log(unscaled / scale + LOG_OFFSET) * sample_data["ref_weights"]) \
                / meta["total_weight"]
            objective += cls._evaluate_penalty(
                feature_weights=feature_weights,
                prototype_weights=prototype_weights,
                lambda_v=lambda_v,
                lambda_w=lambda_w,
                alpha_v=alpha_v,
                alpha_w=alpha_w
            )
        return objective, gradient_tape
