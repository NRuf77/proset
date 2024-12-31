"""Set manager class for prototype set regressor.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np
from scipy import stats

from proset import shared
from proset.set_managers.set_manager import SetManager


class RegressorSetManager(SetManager):
    """Set manager class for proset regressor.
    """

    @staticmethod
    def _get_baseline_distribution(target, weights):
        """Compute baseline distribution parameters from target for regression.

        :param target: 1D numpy array of type specified by shared.FLOAT_TYPE; regression target
        :param weights: see docstring of SetManager.__init__() for details
        :return: dict with keys 'mean' and 'std'; values are floats specifying the weighted mean and standard deviation
            of the target
        """
        mean, std  = shared.check_regressor_target(target=target, weights=weights)
        return {"mean": mean, "std": std}

    @property
    def moments(self):
        """Get marginal mean and standard deviation.

        :return: two float values; mean and standard deviation
        """
        return self._meta["mean"], self._meta["std"]

    @staticmethod
    def _check_batch(batch_info, meta):
        """Check batch definition for consistent dimensions.

        :param batch_info: see docstring of SetManager._check_batch() for details; must have one additional field:
            - target_weight: positive float; inverse bandwidth for kernel on target space
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet
        :return: as return value of SetManager._check_batch()
        """
        shared.check_float_array(x=batch_info["target"], name="batch_info['target']")
        if batch_info["target_weight"] <= 0.0:
            raise ValueError("Parameter target_weight must be strictly positive.")
        return SetManager._check_batch(batch_info=batch_info, meta=meta)

    @staticmethod
    def _process_batch(batch_info):
        """Process batch information.

        :param batch_info: see docstring of SetManager.add_batch() for details
        :return: as return value of SetManager._process_batch() with one additional field:
            - target_weight: positive float; inverse bandwidth for kernel on target space
        """
        batch = SetManager._process_batch(batch_info)
        batch["target_weight"] = batch_info["target_weight"]
        return batch

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
        if prediction_type in [shared.PredictionType.LIKELIHOOD, shared.PredictionType.CDF]:
            if grid is None:
                raise ValueError("Class RegressorSetManager requires parameter grid for prediction type {}.".format(
                    prediction_type.name
                ))
        else:
            if grid is not None:
                raise ValueError(
                    "Class RegressorSetManager does not support parameter grid for prediction type {}.".format(
                        prediction_type.name
                    )
                )
        return SetManager._check_evaluate_input(
            features, num_batches, num_batches_actual, prediction_type, grid, permit_array, meta
        )

    @staticmethod
    def _get_baseline(num_samples, prediction_type, grid, meta):
        """Provide unscaled estimate and scaling for a model with zero batches.

        :param num_samples: see docstring of SetManager._get_baseline() for details
        :param prediction_type: see docstring of SetManager.evaluate_unscaled() for details
        :param grid: see docstring of SetManager.evaluate_unscaled() for details
        :param meta: dict; see docstring of SetManager.evaluate_unscaled() for details; not used by this implementation
        :return: two numpy arrays as a single pair of return values from evaluate_unscaled():
            - depending on prediction type:
              - LIKELIHOOD: 2D array; standard normal density evaluated on the grid
              - CDF: 2D array; standard normal cumulative density evaluated on the grid
              - MEAN: 1D array of zeros; baseline mean
              - MEAN_VAR: 2D array; first column is all zeros, second column is all ones; baseline mean and variance
            - 1D array of ones with the same length as the first dimension of the output
        """
        if prediction_type in [shared.PredictionType.LIKELIHOOD, shared.PredictionType.CDF]:
            if prediction_type == shared.PredictionType.LIKELIHOOD:
                unscaled = stats.norm.pdf(grid).astype(**shared.FLOAT_TYPE)
            else:
                unscaled = stats.norm.cdf(grid).astype(**shared.FLOAT_TYPE)
            if len(grid.shape) == 1:
                unscaled = unscaled[None, :]
                unscaled = np.tile(unscaled, (num_samples, 1))
        elif prediction_type == shared.PredictionType.MEAN:
            unscaled = np.zeros(num_samples, **shared.FLOAT_TYPE)
        else:
            unscaled = np.zeros((num_samples, 2), **shared.FLOAT_TYPE)
            unscaled[:, 1] = 1.0
        return unscaled, np.ones(num_samples, **shared.FLOAT_TYPE)

    @staticmethod
    def _compute_contribution(impact, batch, prediction_type, grid, meta):
        """Compute batch contribution as weighted sum of contributions from kernels on the target space.

        :param impact: 2D array with positive values of type specified by shared.FLOAT_TYPE with one row per sample and
            one column per prototype; prototype impact on each sample
        :param batch: see docstring of SetManager._get_batch_contribution() for details
        :param prediction_type: see docstring of SetManager.evaluate_unscaled() for details
        :param grid: see docstring of SetManager.evaluate_unscaled() for details
        :param meta: dict; not used by this implemenation
        :return: as first return value of SetManager._get_batch_contribution()
        """
        scale = impact.sum(axis=1)
        if prediction_type in [shared.PredictionType.LIKELIHOOD, shared.PredictionType.CDF]:
            if prediction_type == shared.PredictionType.LIKELIHOOD:
                fun = stats.norm.pdf
            else:
                fun = stats.norm.cdf
            contribution = []
            if len(grid.shape) == 1:
                for g in range(grid.shape[0]):  # loop instead of creating potentially large 3d array
                    q_target = fun((grid[g] - batch["target"]) * batch["target_weight"])
                    contribution.append(np.dot(impact, q_target)[:, None])
            else:
                for g in range(grid.shape[1]):
                    q_target = fun((grid[:, g:(g + 1)] - batch["target"]) * batch["target_weight"])
                    # this is a matrix with one row per sample to be scored and one column per prototype
                    contribution.append((impact * q_target).sum(axis=1))
            contribution = np.hstack(contribution)
        else:  # MEAN or MEAN_VAR
            q_target = batch["target"][:, None]  # mean is the target itself
            if prediction_type == shared.PredictionType.MEAN_VAR:
                q_target = np.hstack([q_target, q_target ** 2.0])
            contribution = np.dot(impact, q_target)
            if prediction_type == shared.PredictionType.MEAN_VAR:
                contribution[:, 1] -= contribution[:, 0] ** 2.0 / scale
                # output is unscaled, but the square of weighted expectation needs to apply scaling twice
        return contribution, scale

    @staticmethod
    def _make_dummy_grid():
        """Create dummy grid argumet for _check_evaluate_input().

        :return: returns an 1D array containing a single zero of type specified by shared.FLOAT_TYPE
        """
        return np.zeros(1, **shared.FLOAT_TYPE)

    @staticmethod
    def _copy_batch(batch):
        """Copy and format batch information for output by get_batches().

        :param batch: dict with batch information; as output of _process_batch()
        :return: dict; a single element from the list returned by get_batches(), not including field 'similarities'
        """
        batch_copy = SetManager._copy_batch(batch)
        batch_copy["target_weight"] = batch["target_weight"]
        return batch_copy
