"""Functions that generate artificial test cases for machine learning.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np


def create_checkerboard(size=8, samples_per_square=100, random_state=None):  # pragma: no cover
    """Create samples for the 'checkerboard' problem.

    :param size: integer greater or equal to 2; number of rows and columns making up the checkerboard
    :param samples_per_square: positive integer; average number of samples per square
    :param random_state: instance of numpy.random.RandomState, integer, or None; a random state is used for sampling,
        while any other argument is used as seed to create a random state
    :return: 2D numpy float array and 1d numpy int array; the first value is the (random) feature matrix; the second
        value is the binary target, which is a deterministic function of the features; the first dimension of each value
        has length equal to samples_per_square * size ** 2
    """
    if size < 2:
        raise ValueError("Parameter size must be at least 2.")
    if samples_per_square <= 0:
        raise ValueError("Parameter samples_per_square must positive.")
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    num_samples = samples_per_square * size ** 2
    features = random_state.uniform(low=0.0, high=1.0, size=(num_samples, 2))
    target = np.int64(np.floor(size * features[:, 0]) + np.floor(size * features[:, 1])) % 2
    return features, target


def create_continuous_xor(size=6, samples_per_orthant=100, random_state=None):  # pragma: no cover
    """Create samples for the 'continuous XOR' problem.

    :param size: integer greater or equal to 1; number of features
    :param samples_per_orthant: positive integer; average number of features per orthant
    :param random_state: instance of numpy.random.RandomState, integer, or None; a random state is used for sampling,
        while any other argument is used as seed to create a random state
    :return: 2D numpy float array and 1d numpy int array; the first value is the (random) feature matrix; the second
        value is the binary target, which is a deterministic function of the features; the first dimension of each value
        has length equal to samples_per_orthant * 2 ** size
    """
    if size < 1:
        raise ValueError("Parameter size must be at least 1.")
    if samples_per_orthant <= 0:
        raise ValueError("Parameter samples_per_orthant must positive.")
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    num_samples = samples_per_orthant * 2 ** size
    features = random_state.uniform(low=-1.0, high=1.0, size=(num_samples, size))
    target = np.int64(np.prod(features, axis=1) >= 0)
    return features, target
