"""Constant definitions for testing class ClassifierSetManager.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np

from proset import shared


TARGET = np.array([0, 1, 0, 2, 1, 2])
_, MARGINALS = np.unique(TARGET, return_counts=True)
MARGINALS = (MARGINALS / np.sum(MARGINALS)).astype(**shared.FLOAT_TYPE)
PROTOTYPES = np.array([
    [1.0, 0.0, 0.0, 2.7],
    [1.0, 0.0, 0.0, 1.8],
    [0.0, 1.0, 0.0, -3.5],
    [0.0, 1.0, 0.0, -1.9],
    [0.0, 0.0, 1.0, 12.0],
    [0.0, 0.0, 1.0, 8.0]
], **shared.FLOAT_TYPE)
FEATURE_WEIGHTS = np.array([0.5, 0.0, 1.5, 0.1], **shared.FLOAT_TYPE)
PROTOTYPE_WEIGHTS = np.array([1.0, 2.0, 0.0, 2.0, 1.0, 0.5], **shared.FLOAT_TYPE)
SAMPLE_INDEX = np.array([4, 7, 11, 15, 27, 40])
REFERENCE = np.array([
    [1.0, 0.0, 0.0, 3.0],
    [0.0, 1.0, 0.0, -2.0],
    [0.0, 0.0, 1.0, 9.5]
], **shared.FLOAT_TYPE)
BATCH_INFO = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": FEATURE_WEIGHTS,
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_NO_FEATURES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": np.zeros_like(FEATURE_WEIGHTS, **shared.FLOAT_TYPE),
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_NO_PROTOTYPES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": FEATURE_WEIGHTS,
    "prototype_weights": np.zeros_like(PROTOTYPE_WEIGHTS, **shared.FLOAT_TYPE),
    "sample_index": SAMPLE_INDEX
}
BATCH_INFO_ALL_FEATURES = {
    "prototypes": PROTOTYPES,
    "target": TARGET,
    "feature_weights": np.ones_like(FEATURE_WEIGHTS, **shared.FLOAT_TYPE),
    "prototype_weights": PROTOTYPE_WEIGHTS,
    "sample_index": SAMPLE_INDEX
}
