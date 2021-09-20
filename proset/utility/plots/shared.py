"""Functions shared by multiple submodules of proset.utility.plots.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np


RANGE_DELTA = 0.05
MIN_MARGIN = 1e-3


def make_plot_range(values, delta=RANGE_DELTA, min_margin=MIN_MARGIN):
    """Compute plot range from vector.

    :param values: 1D numpy float array; values for one plot axis
    :param delta: non-negative float; fraction of value range to add as margin on both sides
    :param min_margin: positive float; minimum margin used if plot range is very small or zero
    :return: 1D numpy float array with 2 elements; minimum and maximum value for plots
    """
    min_value = np.min(values)
    max_value = np.max(values)
    margin = max((max_value - min_value) * delta, min_margin)
    return np.array([min_value - margin, max_value + margin])


def check_plot_range(plot_range, parameter_name):
    """Check plot range for consistency

    :param plot_range: as return value of make_plot_range() or None
    :param parameter_name: string; parameter name to be used in exception messages
    :return: no return value; raises an error on invalid input
    """
    if plot_range is None:
        return
    if len(plot_range.shape) != 1:
        raise ValueError("Parameter {} must be a 1D array.".format(parameter_name))
    if plot_range.shape[0] != 2:
        raise ValueError("Parameter {} must have length 2.".format(parameter_name))
    if plot_range[0] >= plot_range[1]:
        raise ValueError("Parameter {} must contain strictly increasing values.".format(parameter_name))


def compute_extent(x_range, y_range, grid_steps):
    """Compute extent parameter for matplotlib.pyplot.imshow().

    :param x_range: 1D numpy float array with 2 elements; x-axis range for plotting
    :param y_range: 1D numpy float array with 2 elements; y-axis range for plotting
    :param grid_steps: positive integer, number of grid steps in each dimension
    :return: 1D numpy float array with 4 elements; left, right, bottom, and top coordinate such that the centers of the
        corner pixels match the x_range and y-range coordinates
    """
    return np.hstack([
        _compute_extent_axis(axis_range=x_range, grid_steps=grid_steps),
        _compute_extent_axis(axis_range=y_range, grid_steps=grid_steps)
    ])


def _compute_extent_axis(axis_range, grid_steps):
    """Compute extent for matplotlib.pyplot.imshow() along one axis.

    :param axis_range: 1D numpy float array with 2 elements; axis range for plotting
    :param grid_steps: positive integer, number of grid steps in each dimension
    :return: 1D numpy float array with 2 elements
    """
    delta = (axis_range[1] - axis_range[0]) / (2.0 * (grid_steps - 1))
    # the range is covered by grid_steps - 1 pixels with one half of a pixel overlapping on each side; delta is half the
    # pixel width
    return np.array([axis_range[0] - delta, axis_range[1] + delta])
