"""Diagnostic plots for hyperparameter search.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import matplotlib.pyplot as plt
import numpy as np

from proset.utility.fit import FitMode
from proset.utility.plots.shared import make_plot_range


def plot_select_results(result, model_name):
    """Plot scores and selected parameters from select_hyperparameters().

    :param result: dict; as return value of select_hyperparameters()
    :param model_name: string; name of the model used in the plot title
    :return: no return value; plots generated
    """
    plot_created = False
    if result["stage_1"]["fit_mode"] == FitMode.BOTH:
        _plot_select_stage_1(stage_1=result["stage_1"], model_name=model_name)
        plot_created = True
    layout = _choose_layout(result["stage_1"]["fit_mode"], result["stage_2"]["num_batch_grid"].shape[0])
    if layout.count(0) < 3:
        _plot_select_parameters(result=result, layout=layout, model_name=model_name)
        plot_created = True
    if not plot_created:
        print("Model was fitted with fixed hyperparameters, no plots are available")


def _plot_select_stage_1(stage_1, model_name):
    """Create a surface plot for the penalty weights chosen via select_hyperparameters().

    :param stage_1: dict; as field "stage_1" from return value of select_hyperparameters()
    :param model_name: see docstring of plot_select_results() for details
    :return: no return value; plot generated
    """
    levels = np.hstack([np.linspace(min(stage_1["scores"]), stage_1["threshold"], 10), max(stage_1["scores"])])
    best = np.log10(stage_1["lambda_grid"][stage_1["best_index"]])
    selected = np.log10(stage_1["lambda_grid"][stage_1["selected_index"]])
    plt.figure()
    plt.tricontourf(
        np.log10(stage_1["lambda_grid"][:, 0]),
        np.log10(stage_1["lambda_grid"][:, 1]),
        stage_1["scores"],  # do not invert scores as the color scale looks better this way
        levels=levels
    )
    legend = [
        plt.plot(best[0], best[1], "bx", markersize=8, markeredgewidth=2)[0],
        plt.plot(selected[0], selected[1], "r+", markersize=8, markeredgewidth=2)[0]
    ]
    plt.grid(True)
    plt.legend(legend, ["Minimizer", "Selection"])
    plt.suptitle("Hyperparameter search for {}: penalty weights".format(model_name))
    plt.xlabel("Log10(lambda_v)")
    plt.ylabel("Log10(lambda_w)")


# pylint: disable=too-many-return-statements
def _choose_layout(fit_mode, batch_trials):
    """Choose plot layout based on which parameters were subject to optimization.

    :param fit_mode: a value of enum FitMode
    :param batch_trials: positive integer; number of different values tried for number of batches
    :return: list with three non-negative integers; these are subplot indicators for matplotlib controlling placement of
        the plots for lambda_v, lambda_w, and number of batches; 0 is used to suppress a plot
    """
    if batch_trials == 1:
        if fit_mode == FitMode.BOTH:
            return [121, 122, 0]  # create plots for both penalty weights
        if fit_mode == FitMode.LAMBDA_V:
            return [111, 0, 0]  # create plot for lambda_v only
        if fit_mode == FitMode.LAMBDA_W:
            return [0, 111, 0]  # create plot for lambda_w only
        return [0, 0, 0]  # create no plot
    if fit_mode == FitMode.BOTH:
        return [131, 132, 133]  # create all plots
    if fit_mode == FitMode.LAMBDA_V:
        return [121, 0, 122]  # create plots for lambda_v and number of batches
    if fit_mode == FitMode.LAMBDA_W:
        return [0, 121, 122]  # create plots for lambda_w and number of batches
    return [0, 0, 111]  # create plot for number of batches only


def _plot_select_parameters(result, layout, model_name):
    """Create plots for the three parameters chosen via select_hyperparameters().

    :param result: dict; as return value of select_hyperparameters()
    :param layout: as return value of _choose_layout()
    :param model_name: string; name of the model used in the plot title
    :return: no return value; plots generated
    """
    scores = np.hstack([result["stage_1"]["scores"], result["stage_2"]["scores"]])
    scores = scores[np.logical_not(np.isnan(scores))]  # either stage may have been skipped
    y_range = make_plot_range(scores)
    plt.figure()
    if layout[0] > 0:
        plt.subplot(layout[0])
        _plot_search_1d(
            grid=result["stage_1"]["lambda_grid"][:, 0],
            scores=result["stage_1"]["scores"],
            threshold=result["stage_1"]["threshold"],
            selected_parameter=result["stage_1"]["lambda_grid"][result["stage_1"]["selected_index"], 0],
            y_range=y_range,
            x_label="Lambda_v",
            title="Stage 1: selected lambda_v = {:0.1e}",
            do_show_legend=True
        )
        plt.xscale("log")
    if layout[1] > 0:
        plt.subplot(layout[1])
        _plot_search_1d(
            grid=result["stage_1"]["lambda_grid"][:, 1],
            scores=result["stage_1"]["scores"],
            threshold=result["stage_1"]["threshold"],
            selected_parameter=result["stage_1"]["lambda_grid"][result["stage_1"]["selected_index"], 1],
            y_range=y_range,
            x_label="Lambda_w",
            title="Stage 1: selected lambda_w = {:0.1e}",
            do_show_legend=False
        )
        plt.xscale("log")
    if layout[2] > 0:
        plt.subplot(layout[2])
        _plot_search_1d(
            grid=result["stage_2"]["num_batch_grid"],
            scores=result["stage_2"]["scores"],
            threshold=result["stage_2"]["threshold"],
            selected_parameter=result["stage_2"]["num_batch_grid"][result["stage_2"]["selected_index"]],
            y_range=y_range,
            x_label="Number of batches",
            title="Stage 2: selected number of batches = {}",
            do_show_legend=False
        )
    plt.suptitle("Hyperparameter search for {}".format(model_name))


def _plot_search_1d(grid, scores, threshold, selected_parameter, y_range, x_label, title, do_show_legend):
    """Plot results from parameter search for one parameter.

    :param grid: 1D numpy float array; parameter grid; does not have to be sorted
    :param scores: 1D numpy float array; scores corresponding to grid
    :param threshold: float; threshold for parameter selection
    :param selected_parameter: float; selected parameter value
    :param y_range: numpy float array with two values; y-axis range
    :param x_label: string; x-axis label
    :param title: string; sub-plot title with '{}' format specifier for selected value
    :param do_show_legend: boolean; whether to include legend
    :return: no return value; plot generated in current axes
    """
    order = np.argsort(grid)
    grid = grid[order]
    scores = -scores[order]  # display as log-loss minimization, not log-likelihood maximization
    x_range = make_plot_range(grid, delta=0.0, min_margin=0.0)
    threshold = -threshold
    y_range = -y_range[-1::-1]
    legend = [
        plt.plot(grid, scores, linewidth=2, color="k")[0],
        plt.plot(x_range, np.ones(2) * np.min(scores), linewidth=2, color="b", linestyle="--")[0],
        plt.plot(x_range, np.ones(2) * threshold, linewidth=2, color="b")[0],
        plt.plot(np.ones(2) * selected_parameter, y_range, linewidth=2, color="r")[0]
    ]
    plt.grid(True)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(x_label)
    plt.ylabel("Log-loss")
    plt.title(title.format(selected_parameter))
    if do_show_legend:
        plt.legend(legend, ["Mean score", "Best score", "Best score + 1 SE", "Selected parameter"], loc="upper right")
