"""Diagnostic plots for hyperparameter search.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import matplotlib.pyplot as plt
import numpy as np

from proset.utility.plots.shared import make_plot_range


GREY = (0.5, 0.5, 0.5)


def plot_select_results(result, model_name):
    """Plot results of hyperparameter selection.

    :param result: dict; as return value of select_hyperparameters()
    :param model_name: string; name of the model used in the plot title
    :return: no return value; plots generated
    """
    cv_results = result["search"]["cv_results"]
    lambda_v_grid = np.unique(cv_results["lambda_v"].values)
    num_batch_grid = np.unique(cv_results["num_batches"].values)
    best_lambda_v = cv_results["lambda_v"].iloc[result["search"]["best_ix"]]
    best_batches = cv_results["num_batches"].iloc[result["search"]["best_ix"]]
    best_score = -cv_results["mean_score"].iloc[result["search"]["best_ix"]]
    # model is set up to maximize scores, so flip sign to get loss function
    selected_lambda_v = cv_results["lambda_v"].iloc[result["search"]["selected_ix"]]
    selected_batches = cv_results["num_batches"].iloc[result["search"]["selected_ix"]]
    selected_score = -cv_results["mean_score"].iloc[result["search"]["selected_ix"]]
    threshold = -result["search"]["threshold"]
    y_range = make_plot_range(values=-cv_results["mean_score"].values)
    plt.figure()
    plt.subplot(121)
    _make_subplot(
        cv_results=cv_results,
        x_key="lambda_v",
        x_grid=lambda_v_grid,
        x_values=(best_lambda_v, selected_lambda_v),
        line_key="num_batches",
        line_grid=num_batch_grid,
        line_values=(best_batches, selected_batches),
        line_format="{:d}",
        y_range=y_range,
        scores=(best_score, selected_score),
        threshold=threshold
    )
    plt.xscale("log")
    plt.subplot(122)
    _make_subplot(
        cv_results=cv_results,
        x_key="num_batches",
        x_grid=num_batch_grid,
        x_values=(best_batches, selected_batches),
        line_key="lambda_v",
        line_grid=lambda_v_grid,
        line_values=(best_lambda_v, selected_lambda_v),
        line_format="{:.2e}",
        y_range=y_range,
        scores=(best_score, selected_score),
        threshold=threshold
    )
    plt.suptitle("Hyperparameter search for {}".format(model_name))


def _make_subplot(
        cv_results,
        x_key,
        x_grid,
        x_values,
        line_key,
        line_grid,
        line_values,
        line_format,
        y_range,
        scores,
        threshold
):
    """Create one plot with parameter search results.

    :param cv_results: pandas data frame; cross-validation results returned by select_hyperparameters() under the key
        'cv_results'
    :param x_key: string; name of x-axis parameter as column name in cv_results
    :param x_grid: 1D numpy float array; x-axis grid
    :param x_values: tuple of two floats; best and selected parameter value on x-axis
    :param line_key: string; name of parameter used to distinguish lines as column name in cv_results
    :param line_grid: 1D numpy float array; parameter values for distinct lines
    :param line_values: tuple of two floats; best and selected parameter value for distinct lines
    :param line_format: string; string format specifier for parameter values used to distinguish lines
    :param y_range: tuple of two floats; y-axis range
    :param scores: tuple of two floats; best and selected log-loss
    :param threshold: float; upper bound on log-loss used to selected model parameters
    :return: no return value; plot generated
    """
    x_range = np.array([x_grid[0], x_grid[-1]])
    handle = None
    for value in line_grid:
        if value not in line_values:
            handle = _plot_curve(data=cv_results[cv_results[line_key] == value], x_key=x_key, color=GREY)
    legend = [
        _plot_curve(data=cv_results[cv_results[line_key] == line_values[0]], x_key=x_key, color="r"),
        _plot_curve(data=cv_results[cv_results[line_key] == line_values[1]], x_key=x_key, color="b")
    ]
    legend_text = [
        "Best solution, {} = {}".format(line_key, line_format.format(line_values[0])),
        "Selected solution, {} = {}".format(line_key, line_format.format(line_values[1]))
    ]
    if handle is not None:
        legend.append(handle)
        legend_text.append("Other solutions")
    legend.extend([
        _set_marker(marker_x=x_values[0], marker_y=scores[0], face_color="r"),
        plt.plot(x_range, [threshold, threshold], linewidth=2, color="r", linestyle="--")[0],
        _set_marker(marker_x=x_values[1], marker_y=scores[1], face_color="b")
    ])
    legend_text.extend(["Best score", "Threshold", "Selected score"])
    plt.grid(True)
    plt.gca().set_xticks(x_grid)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(x_key)
    plt.ylabel("Log-loss")
    plt.legend(legend, legend_text, loc="upper center")


def _plot_curve(data, x_key, color):
    """Plot results for one value of lambda_v across all batches.

    :param data: dict; as return value of select_hyperparameters() limited to one value of lambda_v
    :param x_key: see docstring of _make_subplot() for details
    :param color: string or tuple; matplotlib color definition
    :return: matplotlib line handle
    """
    return plt.plot(data[x_key].values, -data["mean_score"].values, linewidth=2, color=color)[0]


def _set_marker(marker_x, marker_y, face_color):
    """Place a marker on the plot.

    :param marker_x: float; marker x-coordinate
    :param marker_y: float; marker y-coordinate
    :param face_color: string or tuple; matplotlib color definition
    :return: matplotlib line handle
    """
    return plt.plot(
        [marker_x, marker_x],
        [marker_y, marker_y],
        linewidth=0,
        markersize=10,
        markeredgewidth=1,
        marker="X",
        markeredgecolor="k",
        markerfacecolor=face_color
    )[0]
