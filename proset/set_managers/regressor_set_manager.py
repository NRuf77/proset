"""Set manager class for prototype set regressor.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np

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
        # noinspection PyUnresolvedReferences
        return self._meta["mean"], self._meta["std"]


    @staticmethod
    def _check_batch(batch_info, meta):
        """Check batch definition for consistent dimensions.

        :param batch_info: see docstring of SetManager._check_batch() for details
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet
        :return: as return value of SetManager._check_batch()
        """
        shared.check_float_array(x=batch_info["target"], name="batch_info['target']")
        return SetManager._check_batch(batch_info=batch_info, meta=meta)
