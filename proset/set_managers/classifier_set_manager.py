"""Set manager class for prototype set classifier.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np

from proset import shared
from proset.set_managers.set_manager import SetManager


class ClassifierSetManager(SetManager):
    """Set manager class for proset classifier.
    """

    @staticmethod
    def _get_baseline_distribution(target, weights):
        """Compute baseline distribution parameters from target for classification.

        :param target: 1D numpy integer array; class labels encoded as integers from 0 to K - 1
        :param weights: see docstring of SetManager.__init__() for details
        :return: dict with key 'marginals' containing a 1D numpy array of type specified by shared.FLOAT_TYPE with the
            marginal distribution of the classes
        """
        counts = shared.check_classifier_target(target=target, weights=weights)
        return {"marginals": (counts / np.sum(counts))}

    @property
    def marginals(self):
        """Get marginal probabilities.

        :return: 1D numpy array of type specified by shared.FLOAT_TYPE; marginal class probabilities as values in
            (0.0, 1.0)
        """
        # noinspection PyUnresolvedReferences
        return self._meta["marginals"].copy()

    @staticmethod
    def _check_batch(batch_info, meta):
        """Check batch definition for consistent dimensions.

        :param batch_info: see docstring of SetManager.add_batch() for details
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet; must have key
            'marginals' containing a 1D numpy array of type specified by shared.FLOAT_TYPE with the marginal
            distribution of the classes
        :return: as return value of SetManager._check_batch()
        """
        if not np.issubdtype(batch_info["target"].dtype, np.integer):
            raise TypeError("Parameter target must have integer elements.")
        if np.any(batch_info["target"] < 0) or np.any(batch_info["target"] >= meta["marginals"].shape[0]):
            raise ValueError("Parameter target must encode the classes as integers from 0 to K - 1.")
        return SetManager._check_batch(batch_info=batch_info, meta=meta)

    @classmethod
    def _check_evaluate_input(
        cls, features, num_batches, num_batches_actual, prediction_type, grid, permit_array, meta
    ):
        """Check whether input to evaluate_unscaled() is consistent.

        :param features: see docstring of SetManager.evaluate_unscaled() for details
        :param num_batches: see docstring of SetManager.evaluate_unscaled() for details
        :param num_batches_actual: see docstring of SetManager.evaluate_unscaled() for details
        :param prediction_type: see docstring of SetManager.evaluate_unscaled() for details
        :param grid: see docstring of SetManager.evaluate_unscaled() for details
        :param permit_array: see docstring of SetManager.evaluate_unscaled() for details
        :param meta: see docstring of SetManager.evaluate_unscaled() for details
        :return: as SetManager._check_evaluate_input()
        """
        if prediction_type != shared.PredictionType.LIKELIHOOD:
            raise ValueError(
                "Class ClassifierSetManager does not support prediction type {}.".format(prediction_type.name)
            )
        if grid is not None:
            raise ValueError("Class ClassifierSetManager does not support evaluation on a grid.")
        return SetManager._check_evaluate_input(
            features, num_batches, num_batches_actual, prediction_type, grid, permit_array, meta
        )

    # pylint: disable=unused-argument
    @staticmethod
    def _get_baseline(num_samples, prediction_type, grid, meta):
        """Provide unscaled estimate and scaling for a model with zero batches.

        :param num_samples: see docstring of SetManager._get_baseline() for details
        :param prediction_type: see docstring of SetManager.evaluate_unscaled() for details; not used by this
            implementation
        :param grid: see docstring of SetManager.evaluate_unscaled() for details; not used by this implementation
        :param meta: dict; must have key 'marginals' referencing the marginal distribution of classes
        :return: two numpy arrays as a single pair of return values from evaluate_unscaled(); unscaled predictions from
            ClassifierSetManager have a 2D array in first place
        """
        return np.tile(meta["marginals"], (num_samples, 1)).astype(**shared.FLOAT_TYPE), \
            np.ones(num_samples, **shared.FLOAT_TYPE)

    @staticmethod
    def _compute_contribution(impact, batch, prediction_type, grid, meta):
        """Compute batch contribution as weighted sum of contributions from kernels on the target space.

        :param impact: 2D array with positive values of type specified by shared.FLOAT_TYPE with one row per sample and
            one column per prototype; prototype impact on each sample
        :param batch: see docstring of SetManager._get_batch_contribution() for details
        :param prediction_type: see docstring of SetManager.evaluate_unscaled() for details
        :param meta: dict; must have key 'marginals' referencing 1D numpy array of type specified by shared.FLOAT_TYPE
            and values in (0.0, 1.0); marginal distributions of the classes
        :return: as return value of SetManager._get_batch_contribution()
        """
        sort_ix = np.argsort(batch["target"])
        # np.add.reduceat() requires prototypes with the same target value grouped together
        changes = shared.find_changes(batch["target"][sort_ix])
        contribution = np.zeros((impact.shape[0], meta["marginals"].shape[0]), **shared.FLOAT_TYPE)
        contribution[:, np.unique(batch["target"])] = np.add.reduceat(impact[:, sort_ix], indices=changes, axis=1)
        return contribution, np.sum(contribution, axis=1)
