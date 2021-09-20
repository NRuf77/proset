"""Exploratory plots for proset models: concrete subclass implementation for classifiers.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import is_classifier
from sklearn.neighbors import KernelDensity

from proset.utility.plots.explore import ModelPlots, GRID_STEPS, SMALL_MARKER_SIZE, MEDIUM_MARKER_SIZE, \
    LARGE_MARKER_SIZE
from proset.utility.plots.shared import RANGE_DELTA


BRIGHT_COLORS = (  # red used second as class 1 is often the critical one by convention for binary classification
    np.array([0.3, 1.0, 0.3]),
    np.array([1.0, 0.3, 0.3]),
    np.array([0.3, 0.3, 1.0])
)
DARK_COLORS = (
    np.array([0.0, 0.8, 0.0]),
    np.array([0.8, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.8])
)
CLASS_COLORS = np.array(["g", "r", "b", "k", "y", "c", "m"])
LARGE_CLASS_SYMBOLS = np.array(["o", "P", "X"])
SMALL_CLASS_SYMBOLS = np.array([".", "+", "x"])


class ClassifierPlots(ModelPlots):
    """Proset classifier model plots.
    """

    @staticmethod
    def _check_model(model):
        """Check that the model is suitable to use with a given subclass.

        :param model: see docstring of ModelPlots.__init__() for details
        :return: no return value; raises an error if the model is not suitable
        """
        if not is_classifier(model):
            raise ValueError("Parameter model must be an sklearn classifier for class ClassifierPlots.")

    @classmethod
    def _compute_colors(cls, model, grid, compute_familiarity, **kwargs):
        """Compute colors and familiarity for surface plot.

        :param model: a fitted sklearn classifier with at most 3 classes
        :param grid: 2d numpy float array; feature matrix representing surface grid
        :param compute_familiarity: boolean; whether to compute familiarity on the grid
        :param kwargs: this function takes one optional keyword argument:
            - use_proba: boolean; whether to create a probability surface (colors represent the predicted probability
              vector) or a decision surface (colors represent the predicted class); defaults to False if not provided
        :return: three return values:
            - as first return value of ModelPlots._compute_surface()
            - 1D numpy array or None; familiarity for the grid points; only computed if compute_familiarity is True
            - as third return value of ModelPlots._compute_surface()
        """
        if len(model.classes_) > 3:
            raise RuntimeError(
                "Function ClassifierPlots.plot_surface() does not support classifiers with more than 3 classes."
            )
        use_proba = kwargs["use_proba"] if "use_proba" in kwargs.keys() else False
        if use_proba:
            plot_settings = {"plot_type": "Probability surface", "interpolation": "lanczos"}
            if compute_familiarity:
                prediction = model.predict_proba(grid, compute_familiarity=True)  # this relies on proset model features
            else:
                prediction = model.predict_proba(grid)
        else:
            plot_settings = {"plot_type": "Decision surface", "interpolation": None}
            if compute_familiarity:
                prediction = model.predict(grid, compute_familiarity=True)  # this relies on proset model features
            else:
                prediction = model.predict(grid)
        if compute_familiarity:
            grid_familiarity = prediction[1]
            prediction = prediction[0]
        else:
            grid_familiarity = None
        if use_proba:
            grid_values = prediction
        else:
            prediction = cls._encode_target(model=model, target=prediction)
            grid_values = np.zeros((prediction.shape[0], model.classes_.shape[0]))
            grid_values[np.arange(prediction.shape[0]), prediction] = 1.0
        colors = np.outer(grid_values[:, 0], BRIGHT_COLORS[0])
        for i in range(1, model.classes_.shape[0]):
            colors += np.outer(grid_values[:, i], BRIGHT_COLORS[i])
        colors = np.reshape(colors, (GRID_STEPS, GRID_STEPS, 3))
        return colors, grid_familiarity, plot_settings

    @staticmethod
    def _encode_target(model, target):
        """Convert target vector to integer.

        :param model: fitted sklearn classifier
        :param target: 1D numpy array; predicted classes
        :return: 1D numpy array of non-negative integers; integer classes
        """
        return np.searchsorted(model.classes_, target)

    @classmethod
    def _get_surface_target_color(cls, model, target):
        """Determine target plot color for the surface plot.

        :param model: see docstring of ModelPlots.__init__() for details
        :param target: see docstring of ModelPlots.plot_surface() for details
        :return: list of marker face colors for matplotlib.plot()
        """
        return [DARK_COLORS[i] for i in cls._encode_target(model=model, target=target)]

    @classmethod
    def _plot_surface_samples(cls, model, features, target):
        """Add samples positions to surface plot.

        :param model: a fitted sklearn classifier
        :param features: 2D numpy float array with two columns
        :param target: see docstring of ModelPlots.plot_surface() for details
        :return: two return values
            - list of matplotlib plot handles to create a legend for
            - list of strings; legend text for each plot handle
        """
        legend = []
        legend_text = []
        for label in model.classes_:
            color = cls._get_surface_target_color(model=model, target=np.array([label]))[0]
            if target is not None:
                keep_ix = target == label
                plt.plot(
                    features[keep_ix, 0],
                    features[keep_ix, 1],
                    linestyle="",
                    marker="o",
                    markeredgewidth=1,
                    markeredgecolor="k",
                    markersize=SMALL_MARKER_SIZE,
                    markerfacecolor=color
                )
            legend.append(plt.plot(  # create empty plot for legend entry
                np.array([]),
                np.array([]),
                linestyle="",
                marker="o",
                markeredgewidth=1,
                markeredgecolor="k",
                markersize=SMALL_MARKER_SIZE,
                markerfacecolor=color
            )[0])
            legend_text.append(str(label))
        return legend, legend_text

    @classmethod
    def _create_scatter_plot(cls, model, features, target, from_report, marker_size, alpha):
        """Create scatter plot.

        :param model: a fitted proset classifier
        :param features: 2D numpy float array with two columns
        :param target: see docstring of ModelPlots.plot_batch_map() for details
        :param from_report: boolean; whether the target value is taken from the model report or function input
        :param marker_size: 1D numpy array of positive floats; squared marker size for scatter plot
        :param alpha: see docstring of __init__() for details
        :return: two return values:
            - list of plot handles for creating a matplotlib legend
            - list of strings; legend text
        """
        if not from_report:
            target = cls._encode_target(model=model, target=target)
        else:
            target = target.astype(int)  # report values may be floats which break the index computation
        labels = np.unique(target)
        legend = []
        for label in labels:
            color, marker = cls._get_marker_properties(label=label, symbol_set=LARGE_CLASS_SYMBOLS)
            keep_ix = target == label
            plt.scatter(
                x=features[keep_ix, 0],
                y=features[keep_ix, 1],
                s=marker_size[keep_ix],
                c=color,
                marker=marker,
                alpha=alpha
            )
            legend.append(plt.scatter(
                x=np.array([]),
                y=np.array([]),
                s=LARGE_MARKER_SIZE ** 2.0,
                c=color,
                marker=marker,
                alpha=alpha
            ))  # empty plot to get consistent marker size in legend
        return legend, [str(label) for label in model.classes_]

    @staticmethod
    def _get_marker_properties(label, symbol_set):
        """Get scatter plot marker shape and color index for one class.

        :param label: non-negative integer
        :param symbol_set: list of strings or None; matplotlib symbol list
        :return: two strings, matplotlib color and marker type; the second return value is None if symbol_set is None
        """
        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        if symbol_set is None:
            return color, None
        return color, symbol_set[(label // len(CLASS_COLORS)) % len(symbol_set)]

    @classmethod
    def _get_map_target_color(cls, model, target):
        """Determine target plot color for the map plot.

        :param model: fitted sklearn classifier
        :param target: see docstring of ModelPlots.plot_batch_map() for details
        :return: list of marker face colors for matplotlib.plot()
        """
        labels = cls._encode_target(model=model, target=target)
        return [cls._get_marker_properties(label=label, symbol_set=None)[0] for label in labels]

    @classmethod
    def _create_density_plot(
            cls,
            model,
            alpha,
            feature,
            sample_weights,
            target,
            index,
            bandwidth,
            is_prototypes,
            highlight,
            highlight_name,
            x_range,
            y_max
    ):
        """Create density plot for one feature.

        :param model: fitted proset classifier
        :param alpha: see docstring of ModelPlots.__init__() for details
        :param feature: 1D numpy float array; prototype feature values for one dimension
        :param sample_weights: 2D numpy float array with a single column; prototype values along one dimension
        :param target: see docstring of plot_features() for details
        :param index: 1D numpy integer array; sample index to show in plot; pass None to show no index; this parameter
            is ony used if is_prototypes is True
        :param bandwidth: positive float; bandwidth for kernel density estimator
        :param is_prototypes: boolean; whether the samples represent prototypes or supplementary features
        :param highlight: see docstring of plot_features() for details; this parameter is only used if is_prototypes is
            False
        :param highlight_name: see docstring of plot_features() for details; this parameter is only used if
            is_prototypes is False
        :param x_range: 1D numpy float array with two numbers; plot range for x-axis
        :param y_max: non-negative float; y-axis upper bound from previous call; pass 0.0 if no previous value
        :return: four return values:
            - list of matplotlib plot handles; plots to create legend for
            - list of strings; corresponding legend text
            - 1D numpy float array with two numbers; plot range for y-axis
            - float; yaxis value for plotting reference point and point to be explained
        """
        if is_prototypes:
            line_style = "-"
            target = target.astype(int)
        else:
            line_style = "--"
            target = cls._encode_target(model=model, target=target)
        labels = np.unique(target)
        range_factor = (1.0 + RANGE_DELTA)
        y_max /= range_factor  # convert to exact maximum from previous call
        kde = KernelDensity(bandwidth=bandwidth)
        grid = np.reshape(np.linspace(start=x_range[0], stop=x_range[1], num=GRID_STEPS), (GRID_STEPS, 1))
        legend = []
        label_ix = []
        label_features = []
        for label in labels:
            color = cls._get_marker_properties(label=label, symbol_set=None)[0]
            label_ix.append(target == label)
            label_features.append(np.reshape(feature[label_ix[-1]], (np.sum(label_ix[-1]), 1)))
            kde.fit(
                X=label_features[-1],
                sample_weight=sample_weights[label_ix[-1]] if sample_weights is not None else None
            )
            density = np.exp(kde.score_samples(grid))
            y_max = max(y_max, np.max(density))
            legend.append(plt.plot(grid, density, color + line_style, linewidth=2, alpha=alpha)[0])
            if is_prototypes:
                vertical = np.exp(kde.score_samples(label_features[-1]))
                for i in range(vertical.shape[0]):
                    plt.plot([label_features[-1][i], label_features[-1][i]], [0.0, vertical[i]], color, alpha=alpha)
                if index is not None:
                    text_features = np.hstack([
                        label_features[-1],
                        np.reshape(np.exp(kde.score_samples(label_features[-1])), (np.sum(label_ix[-1]), 1))
                    ])
                    cls._print_index(
                        features=text_features,
                        index=index[label_ix[-1]],
                        x_range=x_range,
                        y_range=np.zeros(2)
                    )
        y_step = y_max / (labels.shape[0] + 2.0)
        handle = None
        if not is_prototypes:  # do this after the loop so the final y_max is available
            for label in labels:
                color = cls._get_marker_properties(label=label, symbol_set=None)[0]
                plot_features = np.hstack([
                    label_features[label],
                    (label + 1.0) * y_step * np.ones((label_features[label].shape[0], 1))
                ])
                plt.plot(plot_features[:, 0], plot_features[:, 1], color + ".", alpha=alpha)
                if highlight is not None:
                    handle = cls._plot_highlights(
                        features=plot_features,
                        highlight=highlight[label_ix[label]],
                        marker_size=MEDIUM_MARKER_SIZE
                    )
        legend_text = [str(label) for label in model.classes_]
        if highlight is not None:
            legend.append(handle)
            legend_text.append(highlight_name)
        return legend, legend_text, np.array([0.0, y_max * range_factor]), (labels.shape[0] + 1.0) * y_step
