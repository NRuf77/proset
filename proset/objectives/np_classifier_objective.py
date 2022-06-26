"""Class for proset classifier objective function that uses numpy to evaluate the likelihood.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np
import scipy.linalg.blas as blas  # pylint: disable=no-name-in-module

from proset.objectives.np_objective import NpObjective
import proset.objectives.shared_classifier as shared_classifier
import proset.shared as shared


class NpClassifierObjective(NpObjective):
    """Objective function for a proset classifier that uses numpy to evaluate the likelihood.
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
        meta = NpObjective._check_init_values(
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

        :param candidates: see docstring of np_objective.NpObjective._finalize_split() for details
        :param features: see docstring of np_objective.NpObjective._finalize_split() for details
        :param target: see docstring of _check_init_values() for details
        :param weights: see docstring of np_objective.NpObjective._finalize_split() for details
        :param unscaled: see docstring of np_objective.NpObjective._finalize_split() for details
        :param scale: see docstring of np_objective.NpObjective._finalize_split() for details
        :param meta: dict; must have key 'num_classes' referencing the number of classes
        :return: dict in the format specified for the return value of np_objective.NpObjective._split_samples(); this
            function reorders the candidate information such that the target values are in ascending order; it adds the
            following fields:
            - cand_changes: 1D numpy integer array; index vector indicating changes in the candidate target, including
              zero for the first element
            - class_matches: 2D numpy boolean array with one row per reference point and one column per prototype
              candidate; indicates whether the respective reference and candidate point have the same target value
        """
        sample_data = NpObjective._finalize_split(
            candidates=candidates,
            features=features,
            target=target,
            weights=weights,
            unscaled=unscaled,
            scale=scale,
            meta=meta
        )
        sample_data["ref_weights"] = shared_classifier.adjust_ref_weights(
            sample_data=sample_data, candidates=candidates, target=target, weights=weights, meta=meta
        )
        order = np.argsort(sample_data["cand_target"])
        sample_data["cand_features"] = sample_data["cand_features"][order].astype(**shared.FLOAT_TYPE)
        # ensure 2D arrays have specified order
        sample_data["cand_features_squared"] = sample_data["cand_features_squared"][order].astype(**shared.FLOAT_TYPE)
        sample_data["cand_target"] = sample_data["cand_target"][order]
        sample_data["cand_changes"] = shared.find_changes(sample_data["cand_target"])
        sample_data["cand_index"] = sample_data["cand_index"][order]
        sample_data["class_matches"] = shared_classifier.find_class_matches(sample_data=sample_data, meta=meta)
        return sample_data

    @classmethod
    def _evaluate_likelihood(cls, feature_weights, prototype_weights, sample_data, similarity, meta):
        """Compute negative log-likelihood function and gradient without the penalty term.

        :param feature_weights: see docstring of np_objective.NpObjective._evaluate_likelihood() for details
        :param prototype_weights: see docstring of np_objective.NpObjective._evaluate_likelihood() for details
        :param sample_data: see docstring of np_objective.NpObjective._evaluate_likelihood() for details
        :param similarity: see docstring of np_objective.NpObjective._evaluate_likelihood() for details
        :param meta: dict; must have key 'total_weight' referencing the total sample weight
        :return: as return values of np_objective.NpObjective._evaluate_likelihood()
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

        :param similarity: as return value of np_objective.NpObjective._compute_similarity()
        :param sample_data: as return value of objective.Objective._split_samples() after processing by
            objective.Objective._shrink_sample_data()
        :param prototype_weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE;
            prototype weights
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
            - ref_unscaled: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; unscaled
              probability estimates for the true class of each reference point
            - ref_scale: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; scales
              corresponding to ref_unscaled
            - ref_weights: sample_data["ref_weights"]
            - total_weight: meta["total_weight"]
        """
        similarity_matched = similarity * sample_data["class_matches"]
        impact = similarity * prototype_weights
        ref_unscaled = sample_data["ref_unscaled"].copy()
        # noinspection PyUnresolvedReferences
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
        return -1.0 * blas.sdot(  # pylint: disable=no-member
            x=np.log(shared_expressions["ref_unscaled"] / shared_expressions["ref_scale"] + shared.LOG_OFFSET),
            y=shared_expressions["ref_weights"]
        ) / shared_expressions["total_weight"]

    @classmethod
    def _compute_partial_feature_weights(cls, shared_expressions, feature_weights):
        """Compute the vector of partial derivatives w.r.t. the feature weights.

        :param shared_expressions: as return value of _compute_shared_expressions()
        :param feature_weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; feature
            weights
        :return: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE; partial derivatives
            w.r.t. feature weights
        """
        if feature_weights.shape[0] == 0:
            # safeguard in case optimization for sparseness removes all features; functions from scipy.linalg.blas do
            # not handle zero dimensions gracefully
            return np.zeros(0, **shared.FLOAT_TYPE)
        part_1 = blas.sgemv(  # pylint: disable=no-member
            alpha=1.0,
            a=cls._quick_compute_part(
                ref_features=shared_expressions["ref_features"],
                ref_features_squared=shared_expressions["ref_features_squared"],
                cand_features=shared_expressions["cand_features"],
                cand_features_squared=shared_expressions["cand_features_squared"],
                impact=shared_expressions["impact"]
            ),
            x=shared_expressions["ref_weights"] / shared_expressions["ref_scale"],
            trans=1
        )
        part_2 = blas.sgemv(  # pylint: disable=no-member
            alpha=1.0,
            a=cls._quick_compute_part(
                ref_features=shared_expressions["ref_features"],
                ref_features_squared=shared_expressions["ref_features_squared"],
                cand_features=shared_expressions["cand_features"],
                cand_features_squared=shared_expressions["cand_features_squared"],
                impact=shared_expressions["impact_matched"]
            ),
            x=shared_expressions["ref_weights"] / shared_expressions["ref_unscaled"],
            trans=1
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
        :return: 2D numpy array of type specified by shared.FLOAT_TYPE with one row per reference point and one column
            per feature; contribution of squared differences between features for reference and candidate points,
            weighted with impact
        """
        part = (ref_features_squared.transpose() * np.sum(impact, axis=1)).transpose()
        part -= 2.0 * ref_features * blas.sgemm(alpha=1.0, a=impact, b=cand_features)  # pylint: disable=no-member
        return part + blas.sgemm(alpha=1.0, a=impact, b=cand_features_squared)  # pylint: disable=no-member

    @staticmethod
    def _compute_partial_prototype_weights(shared_expressions):
        """Compute the vector of partial derivatives w.r.t. the prototype weights.

        :param shared_expressions: as return value of _compute_shared_expressions()
        :return: 1D numpy array of type specified by shared.FLOAT_TYPE; partial derivatives w.r.t. prototype weights
        """
        partial_prototype_weights = \
            shared_expressions["similarity_matched"].transpose() / shared_expressions["ref_unscaled"]
        partial_prototype_weights -= shared_expressions["similarity"].transpose() / shared_expressions["ref_scale"]
        return -1.0 * blas.sgemv(  # pylint: disable=no-member
            alpha=1.0, a=partial_prototype_weights, x=shared_expressions["ref_weights"]
        ) / shared_expressions["total_weight"]
