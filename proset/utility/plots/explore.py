"""Exploratory plots for proset models: abstract base class for all models.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from statsmodels.distributions.empirical_distribution import ECDF

from proset.shared import check_feature_names, check_scale_offset
from proset.utility.plots.shared import make_plot_range, check_plot_range, compute_extent


ERROR_MESSAGE_FEATURES = "Parameter {} must match the number of features the model was trained on (expecting {})."
GRID_STEPS = 200
CONTOUR_LINE_WIDTH = 3
SMALL_MARKER_SIZE = 5.0
MEDIUM_MARKER_SIZE = 8.0
LARGE_MARKER_SIZE = 10.0
TEXT_OFFSET = 0.005


class ModelPlots(metaclass=ABCMeta):
    """Base class for proset model plots.
    """

    def __init__(
            self,
            model,
            model_name,
            feature_names=None,
            scale=None,
            offset=None,
            alpha=0.5,
            jitter_std=0.1,
            random_state=None
    ):
        """Initialize model plots.

        :param model: a fitted sklearn model; some functions of this class only work for proset models
        :param model_name: string; model name to be used in plot titles
        :param feature_names: list of strings or None; must have one element per feature used to train the model; pass
            None to use default names X0, X1, etc.
        :param scale: 1D numpy array of positive floats or None; scale for transforming prototype features back to their
            original values; pass None for no transform
        :param offset: 1D numpy array of floats or None; offset for transforming prototype features back to their
            original values; pass None for no transform
        :param alpha: float in (0.0, 1.0]; alpha value used for scatter plots
        :param jitter_std: non-negative float; standard deviation of jitter for scatter plots in absolute units
        :param random_state: an instance of np.random.RandomState, integer, or None; used to initialize the random
            number generator for jitter
        """
        self._check_model(model)
        feature_names, scale, offset = self._check_parameters(
            model=model,
            feature_names=feature_names,
            scale=scale,
            offset=offset,
            alpha=alpha,
            jitter_std=jitter_std
        )
        self._model = model
        self._model_name = model_name
        self._feature_names = feature_names
        self._scale = scale
        self._offset = offset
        self._alpha = alpha
        self._jitter_std = jitter_std
        self._random_state = np.random.RandomState(random_state)

    @staticmethod
    @abstractmethod
    def _check_model(model):
        """Check that the model is suitable to use with a given subclass.

        :param model: see docstring of __init__() for details
        :return: no return value; raises an error if the model is not suitable
        """
        raise NotImplementedError("Abstract method ModelPlots._check_model() has no default implementation.")

    @staticmethod
    def _check_parameters(model, feature_names, scale, offset, alpha, jitter_std):
        """Check parameters passed to __init__.

        :param model: see docstring of __init__() for details
        :param feature_names: see docstring of __init__() for details
        :param scale: see docstring of __init__() for details
        :param offset: see docstring of __init__() for details
        :param alpha: see docstring of __init__() for details
        :param jitter_std: see docstring of __init__() for details
        :return: three return values:
            - list of strings: feature names as input or defaults if input is None
            - 1D numpy array of positive floats: as input scale or default if None
            - 1D numpy float array: as input offset or default if None
        """
        check_is_fitted(model, attributes="n_features_in_")
        feature_names = check_feature_names(
            num_features=model.n_features_in_,
            feature_names=feature_names,
            active_features=None
        )
        scale, offset = check_scale_offset(num_features=model.n_features_in_, scale=scale, offset=offset)
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("Parameter alpha must lie in (0.0, 1.0].")
        if jitter_std <= 0.0:
            raise ValueError("Parameter jitter_std must be positive.")
        return feature_names, scale, offset

    def plot_surface(
            self,
            features,
            target,
            baseline=None,
            plot_index=np.array([0, 1]),
            comment=None,
            highlight=None,
            highlight_name=None,
            reference=None,
            explain_features=None,
            explain_target=None,
            familiarity=None,
            quantiles=(0.01, 0.05),
            x_range=None,
            y_range=None,
            **kwargs
    ):
        """Plot model prediction as colored surface over two features with fixed values for additional features.

        :param features: 2D numpy float array or None; feature matrix; sparse matrices or infinite/missing values not
            supported; if passing None, x_range and y_range must be provided to determine the extend of the plot
        :param target: list-like object or None; target for supervised learning; must be None if features is None; can
            be None if features is not None in which case samples are not shown in the plot
        :param baseline: 2D numpy float array with one row or None; feature values used to compute predictions except
            for the two features spanning the surface plot; pass None to use a vector of zeros
        :param plot_index: 1D numpy array of non-negative integers; indices of features spanning the surface plot; must
            have length two
        :param comment: string or None; a string will be added after the supertitle in brackets
        :param highlight: 1D numpy boolean array or None; indicator vector of points to highlight; this can only be not
            None if features is not None
        :param highlight_name: string or None; legend label for highlighted points; this must be provided iff highlight
            is not None
        :param reference: 2D numpy float array with one row or None; feature values for reference point; pass None to
            plot no reference point
        :param explain_features: 2D numpy float array with one row or None; feature values for point to be explained;
            pass None to plot no reference point
        :param explain_target: single value or None; target for point to be explained; this must be provided iff
            explain_features is not None
        :param familiarity: 1D numpy array of non-negative floats or None; if not None, reference values for familiarity
            to use for plotting contour lines; this feature is specific to proset models
        :param quantiles: float or tuple of floats in (0.0, 1.0); quantiles of familiarity for which to plot contour
            lines
        :param x_range: 1D numpy float array with 2 strictly increasing values or None; desired plot range in x
            direction; pass None to choose range based on features
        :param y_range: 1D numpy float array with 2 strictly increasing values or None; desired plot range in y
            direction; pass None to choose range based on features
        :param kwargs: subclasses of ModelPlots may support additional keyword arguments
        :return: two numpy 1D float array of length 2; plot ranges in x- and y-direction
        """
        features, target, baseline, highlight, reference, explain_features, explain_target, quantiles, x_range, \
            y_range = self._check_surface_parameters(
                num_features=self._model.n_features_in_,
                scale=self._scale,
                offset=self._offset,
                features=features,
                target=target,
                baseline=baseline,
                plot_index=plot_index,
                highlight=highlight,
                highlight_name=highlight_name,
                reference=reference,
                explain_features=explain_features,
                explain_target=explain_target,
                familiarity=familiarity,
                quantiles=quantiles,
                x_range=x_range,
                y_range=y_range
            )
        colors, grid_familiarity, plot_settings = self._compute_surface(
            model=self._model,
            scale=self._scale,
            offset=self._offset,
            baseline=baseline,
            plot_index=plot_index,
            familiarity=familiarity,
            x_range=x_range,
            y_range=y_range,
            **kwargs
        )
        self._create_surface_plot(
            model=self._model,
            model_name=self._model_name,
            feature_names=self._feature_names,
            scale=self._scale,
            offset=self._offset,
            features=features,
            target=target,
            plot_index=plot_index,
            comment=comment,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            grid_familiarity=grid_familiarity,
            quantiles=quantiles,
            x_range=x_range,
            y_range=y_range,
            colors=colors,
            plot_settings=plot_settings)
        return x_range, y_range

    @classmethod
    def _check_surface_parameters(
            cls,
            num_features,
            scale,
            offset,
            features,
            target,
            baseline,
            plot_index,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            familiarity,
            quantiles,
            x_range,
            y_range
    ):
        """Check that parameters for plot_surface() are valid.

        :param num_features: positive integer; expected number of features
        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param features: see docstring of plot_surface() for details
        :param target: see docstring of plot_surface() for details
        :param baseline: see docstring of plot_surface() for details
        :param plot_index: see docstring of plot_surface() for details
        :param highlight: see docstring of plot_surface() for details
        :param highlight_name: see docstring of plot_surface() for details
        :param reference: see docstring of plot_surface() for details
        :param explain_features: see docstring of plot_surface() for details
        :param explain_target: see docstring of plot_surface() for details
        :param familiarity: see docstring of plot_surface() for details
        :param quantiles: see docstring of plot_surface() for details
        :param x_range: see docstring of plot_surface() for details
        :param y_range: see docstring of plot_surface() for details
        :return: ten return values: features, target, baseline, highlight, reference, explain_features, explain_target,
            quantiles, x_range, and y-range as proper numpy arrays
        """
        features, target, highlight, reference, explain_features, explain_target = cls._check_plot_parameters(
            num_features=num_features,
            features=features,
            target=target,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            x_range=x_range,
            y_range=y_range
        )
        if baseline is None:
            baseline = np.zeros((1, num_features), dtype=float)
        baseline = check_array(baseline)
        if baseline.shape[0] != 1:
            raise ValueError("Parameter baseline must have a single row.")
        if baseline.shape[1] != num_features:
            raise ValueError(ERROR_MESSAGE_FEATURES.format("baseline", num_features))
        if len(plot_index.shape) != 1:
            raise ValueError("Parameter plot_index must be 1D array.")
        if plot_index.shape[0] != 2:
            raise ValueError("Parameter plot_index must have two elements.")
        if not np.issubdtype(plot_index.dtype, np.integer):
            raise TypeError("Parameter plot_index must be an integer array.")
        if plot_index[0] == plot_index[1]:
            raise ValueError("Parameter plot_index must contain distinct values.")
        if np.any(plot_index < 0):
            raise ValueError("Parameter plot_index must have non-negative elements.")
        if familiarity is not None:
            if len(familiarity.shape) != 1:
                raise ValueError("Parameter familiarity must be 1D array.")
            if np.any(familiarity < 0.0):
                raise ValueError("Parameter familiarity must not have negative elements.")
        quantiles = np.atleast_1d(quantiles)
        if np.any(quantiles <= 0.0) or np.any(quantiles >= 1.0):
            raise ValueError("Parameter quantiles must contain values in (0.0, 1.0).")
        x_range = cls._determine_plot_range(
            x_range,
            features[:, plot_index[0]] * scale[plot_index[0]] + offset[plot_index[0]]
        )  # plot uses original feature range if scale and offset are provided
        y_range = cls._determine_plot_range(
            y_range,
            features[:, plot_index[1]] * scale[plot_index[1]] + offset[plot_index[1]]
        )
        return features, target, baseline, highlight, reference, explain_features, explain_target, quantiles, x_range, \
            y_range

    # pylint: disable=too-many-branches
    @staticmethod
    def _check_plot_parameters(
            num_features,
            features,
            target,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            x_range,
            y_range
    ):
        """Check that common parameters for plots are valid.

        :param num_features: positive integer; expected number of features
        :param features: see docstring of plot_surface() for details
        :param target: see docstring of plot_surface() for details
        :param highlight: see docstring of plot_surface() for details
        :param highlight_name: see docstring of plot_surface() for details
        :param reference: see docstring of plot_surface() for details
        :param explain_features: see docstring of plot_surface() for details
        :param explain_target: see docstring of plot_surface() for details
        :param x_range: see docstring of plot_surface() for details
        :param y_range: see docstring of plot_surface() for details
        :return: six return values: features, target, highlight, reference, explain_features, and explain_target
            converted to proper numpy arrays
        """
        if features is not None:
            if target is None:
                features = check_array(features)
            else:
                features, target = check_X_y(X=features, y=target)
            if features.shape[1] != num_features:
                raise ValueError(ERROR_MESSAGE_FEATURES.format("features", num_features))
        elif target is not None:
            raise ValueError("Parameter target must be None if features is None.")
        if highlight is not None:
            if features is None:
                raise ValueError("Parameter highlight must be None if features is None.")
            if target is None:
                raise ValueError("Parameter highlight must be None if target is None.")
            highlight = check_array(highlight, ensure_2d=False)
            if len(highlight.shape) != 1:
                raise ValueError("Parameter highlight must be a 1D array.")
            if highlight.shape[0] != features.shape[0]:
                raise ValueError("Parameter highlight must have as many elements as features has rows.")
            if highlight.dtype not in [bool, np.bool]:
                raise TypeError("Parameter highlight must be of boolean type.")
            if highlight_name is None:
                raise ValueError("Parameter highlight_name must be provided if highlight is not None.")
        else:
            if highlight_name is not None:
                raise ValueError("Parameter highlight_name must be None if highlight is None.")
        if reference is not None:
            reference = check_array(reference)
            if reference.shape[0] != 1:
                raise ValueError("Parameter reference_features must have a single row.")
            if reference.shape[1] != num_features:
                raise ValueError(ERROR_MESSAGE_FEATURES.format("reference_features", num_features))
        if explain_features is not None:
            if explain_target is None:
                raise ValueError("Parameter explain_target must not be None if explain_features is not None.")
            explain_features, explain_target = check_X_y(X=explain_features, y=np.array([explain_target]))
        else:
            if explain_target is not None:
                raise ValueError("Parameter explain_target must be None if explain_features is None.")
        check_plot_range(plot_range=x_range, parameter_name="x_range")
        check_plot_range(plot_range=y_range, parameter_name="y_range")
        return features, target, highlight, reference, explain_features, explain_target

    @staticmethod
    def _determine_plot_range(plot_range, *args):
        """Determine plot range for one axis.

        :param plot_range: 1D numpy float array or None; if not None, must have exactly two strictly increasing values
        :param args: any number of 1D numpy float arrays or None values; coordinates for points to be plotted
        :return: if plot_range is not None, return input value; else, determine range from args; raises a ValueError if
            all arguments are None or missing
        """
        if plot_range is not None:
            return plot_range
        vectors = [vector for vector in args if vector is not None]
        if len(vectors) == 0:
            raise ValueError("At least one numeric argument is required to calculate a plot range.")
        return make_plot_range(np.hstack(vectors))

    @classmethod
    def _compute_surface(cls, model, scale, offset, baseline, plot_index, familiarity, x_range, y_range, **kwargs):
        """Determine image colors for surface plot.

        :param model: see docstring of __init__() for details
        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param baseline: see docstring of plot_surface() for details
        :param plot_index: see docstring of plot_surface() for details
        :param familiarity: see docstring of plot_surface() for details
        :param x_range: see docstring of plot_surface() for details
        :param y_range: see docstring of plot_surface() for details
        :param kwargs: see docstring of plot_surface() for details
        :return: three return values:
            - 2D numpy array with one row per row of grid and 3 columns; RGB values for each grid point
            - tuple with three 2D numpy float array of dimension GRID_STEPS * GRID_STEPS or None: quantile values for
              familiarity and mesh grid coordinates; returns None if familiarity is None
            - dict with the following fields:
              - plot_type: string; type of plot displayed in plot supertitle
              - interpolation: interpolation parameter for matplotlib.pyplot.imshow()
        """
        x_grid = np.linspace(x_range[0], x_range[1], GRID_STEPS)
        y_grid = np.linspace(y_range[0], y_range[1], GRID_STEPS)
        grid = np.vstack([
            np.tile((x_grid - offset[plot_index[0]]) / scale[plot_index[0]], GRID_STEPS),
            # plot ranges use original feature scales but model needs transformed features for prediction
            np.repeat((y_grid - offset[plot_index[1]]) / scale[plot_index[1]], GRID_STEPS)
        ]).transpose()
        if model.n_features_in_ > 2:
            extended = np.tile(baseline, (grid.shape[0], 1))
            extended[:, plot_index] = grid
        else:
            extended = grid
        colors, grid_familiarity, plot_settings = cls._compute_colors(
            model=model,
            grid=extended,
            compute_familiarity=familiarity is not None,
            **kwargs
        )
        if familiarity is not None:
            mesh = np.meshgrid(x_grid, y_grid)
            grid_familiarity = (
                np.reshape(ECDF(familiarity)(grid_familiarity), (GRID_STEPS, GRID_STEPS)),
                mesh[0],
                mesh[1]
            )
        return colors, grid_familiarity, plot_settings

    @classmethod
    @abstractmethod
    def _compute_colors(cls, model, grid, compute_familiarity, **kwargs):
        """Compute colors and familiarity for surface plot.

        :param model: a fitted sklearn classifier with at most 3 classes
        :param grid: 2d numpy float array; feature matrix representing surface grid
        :param compute_familiarity: boolean; whether to compute familiarity on the grid
        :param kwargs: subclasses of ModelPlots may support additional keyword arguments
        :return: three return values:
            - as first return value of ModelPlots._compute_surface()
            - 1D numpy array or None; familiarity for the grid points; only computed if compute_familiarity is True
            - as third return value of ModelPlots._compute_surface()
        """
        raise NotImplementedError("Abstract method ModelPlots._compute_colors() has no default implementation.")

    @classmethod
    def _create_surface_plot(
            cls,
            model,
            model_name,
            feature_names,
            scale,
            offset,
            features,
            target,
            plot_index,
            comment,
            reference,
            highlight,
            highlight_name,
            explain_features,
            explain_target,
            grid_familiarity,
            quantiles,
            x_range,
            y_range,
            colors,
            plot_settings
    ):
        """Create matplotlib figure with surface plot.

        :param model: see docstring of __init__() for details
        :param model_name: see docstring of __init__() for details
        :param feature_names: see docstring of __init__() for details; None not allowed
        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param features: see docstring of plot_surface() for details
        :param target: see docstring of plot_surface() for details
        :param plot_index: 1D numpy array of non-negative integers; indices of features spanning the surface plot
        :param comment: see docstring of plot_surface() for details
        :param highlight: see docstring of plot_surface() for details
        :param highlight_name: see docstring of plot_surface() for details
        :param reference: see docstring of plot_surface() for details
        :param explain_features: see docstring of plot_surface() for details
        :param explain_target: see docstring of plot_surface() for details
        :param grid_familiarity: as second return value of _compute_surface()
        :param quantiles: see docstring of plot_surface() for details
        :param x_range: see docstring of plot_surface() for details
        :param y_range: see docstring of plot_surface() for details
        :param colors: as first return value of _compute_surface()
        :param plot_settings: as third return value of _compute_surface()
        :return: no return value; figure created
        """
        feature_names, features, reference, explain_features = cls._reduce_features(
            scale=scale,
            offset=offset,
            feature_names=feature_names,
            features=features,
            plot_index=plot_index,
            reference=reference,
            explain_features=explain_features
        )
        plt.figure()
        plt.imshow(
            X=colors,
            interpolation=plot_settings["interpolation"],
            origin="lower",
            extent=compute_extent(x_range=x_range, y_range=y_range, grid_steps=GRID_STEPS)
        )
        if grid_familiarity is not None:
            plt.contour(
                grid_familiarity[1],
                grid_familiarity[2],
                grid_familiarity[0],
                levels=quantiles,
                linewidths=CONTOUR_LINE_WIDTH
            )
        legend, legend_text = cls._plot_surface_samples(
            model=model,
            features=features,
            target=target
        )
        if highlight is not None:
            legend.append(cls._plot_highlights(features=features, highlight=highlight, marker_size=MEDIUM_MARKER_SIZE))
            legend_text.append(highlight_name)
        if reference is not None:
            legend.append(cls._plot_single_point(features=reference, color="none"))
            legend_text.append("reference")
        if explain_features is not None:
            legend.append(cls._plot_single_point(
                features=explain_features,
                color=cls._get_surface_target_color(model=model, target=explain_target)[0])
            )
            legend_text.append("to explain")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.gca().set_aspect(0.66 * np.diff(x_range) / np.diff(y_range))
        plt.grid("on")
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        if len(legend) > 0:
            plt.legend(legend, legend_text)
        title = "{} for {}".format(plot_settings["plot_type"], model_name)
        if comment is not None:
            title += " ({})".format(comment)
        plt.suptitle(title)
        if grid_familiarity is not None:
            plt.title("Contour lines for quantiles of familiarity: {}".format(
                ", ".join(["{:.2f}".format(q) for q in quantiles])
            ))

    @classmethod
    def _reduce_features(cls, scale, offset, feature_names, features, plot_index, reference, explain_features):
        """Apply linear transformation and reduce features to the two dimensions used for plotting

        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param feature_names: see docstring of __init__() for details; None not allowed
        :param features: see docstring of plot_surface() for details
        :param plot_index: see docstring of plot_surface() for details
        :param reference: see docstring of plot_surface() for details
        :param explain_features: see docstring of plot_surface() for details
        :return: four return values: feature_names, features, plot_index, reference, and explain_features transformed
            and reduced to the two dimensions used for plotting
        """
        feature_names = [feature_names[i] for i in plot_index]
        features = cls._transform_reduce(features=features, scale=scale, offset=offset, plot_index=plot_index)
        reference = cls._transform_reduce(features=reference, scale=scale, offset=offset, plot_index=plot_index)
        explain_features = cls._transform_reduce(
            features=explain_features,
            scale=scale,
            offset=offset,
            plot_index=plot_index
        )
        return feature_names, features, reference, explain_features

    @staticmethod
    def _transform_reduce(features, scale, offset, plot_index):
        """Transform and reduce features if not None.

        :param features: see docstring of plot_surface() for details
        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param plot_index: see docstring of plot_surface() for details
        :return: features transformed and reduced or None if features is None
        """
        if features is None:
            return None
        return (features * scale + offset)[:, plot_index]

    @staticmethod
    def _plot_highlights(features, highlight, marker_size):
        """Circle points to be highlighted.

        :param features: 2D numpy float array with a 1 row and 2 columns
        :param highlight: see docstring of plot_surface() for details
        :param marker_size: positive integer; marker size
        :return: plot handle to use for creating a legend
        """
        return plt.plot(
            features[highlight, 0],
            features[highlight, 1],
            linestyle="",
            marker="o",
            markeredgecolor="k",
            markersize=marker_size,
            markerfacecolor="none"
        )[0]

    @staticmethod
    def _plot_single_point(features, color):
        """Add a point of special interest to the current plot.

        :param features: 2D numpy float array with a 1 row and 2 columns
        :param color: marker face color for matplotlib.plot()
        :return: plot handle to use for creating a legend
        """
        return plt.plot(
            features[0, 0],
            features[0, 1],
            linestyle="",
            marker="s",
            markeredgewidth=2,
            markeredgecolor="k",
            markersize=LARGE_MARKER_SIZE,
            markerfacecolor=color
        )[0]

    @classmethod
    @abstractmethod
    def _get_surface_target_color(cls, model, target):
        """Determine target plot color for the surface plot.

        :param model: see docstring of __init__() for details
        :param target: see docstring of plot_surface() for details
        :return: list of marker face colors for matplotlib.plot()
        """
        raise NotImplementedError("Abstract method ModelPlots._get_target_color() has no default implementation.")

    @classmethod
    @abstractmethod
    def _plot_surface_samples(cls, model, features, target):
        """Add samples positions to surface plot.

        :param model: see docstring of __init__() for details
        :param features: see docstring of plot_surface() for details; must be reduced to the two features spanning the
            surface
        :param target: see docstring of plot_surface() for details
        :return: two return values
            - list of matplotlib plot handles to create a legend for
            - list of strings; legend text for each plot handle
        """
        raise NotImplementedError("Abstract method ModelPlots._plot_surface_features() has no default implementation.")

    def plot_batch_map(
            self,
            batch,
            features=None,
            target=None,
            comment=None,
            highlight=None,
            highlight_name=None,
            reference=None,
            explain_features=None,
            explain_target=None,
            show_index=True,
            x_range=None,
            y_range=None
    ):
        """Perform weighted PCA for one batch and plot the first two principal components of the prototypes.

        :param batch: positive integer; batch to use
        :param features: 2D numpy float array or None; feature matrix of supplementary points to be plotted; sparse
            matrices or infinite/missing values not supported
        :param target: list-like object or None; target for supervised learning corresponding to features; must be None
            if features is None
        :param comment: string or None; a string will be added after the supertitle in brackets
        :param highlight: 1D numpy boolean array or None; indicator vector of supplementary points to highlight; this
            can only be not None if features is not None
        :param highlight_name: string or None; legend label for highlighted points; this must be provided iff highlight
            is not None
        :param reference: 2D numpy float array with one row or None; feature values for reference point; pass None to
            plot no reference point
        :param explain_features: 2D numpy float array with one row or None; feature values for point to be explained;
            pass None to plot no reference point
        :param explain_target: single value or None; target for point to be explained; this must be provided iff
            explain_features is not None
        :param show_index: boolean; whether to show the corresponding sampled index next to each prototype in the plot
        :param x_range: 1D numpy float array with 2 strictly increasing values or None; desired plot range in x
            direction; pass None to choose range based on prototypes and other points provided
        :param y_range: 1D numpy float array with 2 strictly increasing values or None; desired plot range in y
            direction; pass None to choose range based on prototypes and other points provided
        :return: two numpy 1D float array of length 2; plot ranges in x- and y-direction
        """
        features, target, highlight, reference, explain_features, explain_target = self._check_batch_plot_parameters(
            num_features=self._model.n_features_in_,
            num_batches=self._model.set_manager_.num_batches,
            batch=batch,
            features=features,
            target=target,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            x_range=x_range,
            y_range=y_range
        )
        batch_info = self._get_batch_info(model=self._model, batch=batch, explain_features=explain_features)
        weighted_pca = self._perform_weighted_pca(batch_info=batch_info, random_state=self._random_state)
        features, reference, explain_features, x_range, y_range = self._prepare_map_plot(
            batch_info=batch_info,
            weighted_pca=weighted_pca,
            features=features,
            reference=reference,
            explain_features=explain_features,
            x_range=x_range,
            y_range=y_range
        )
        self._create_map_plot(
            model=self._model,
            model_name=self._model_name,
            alpha=self._alpha,
            batch=batch,
            batch_info=batch_info,
            weighted_pca=weighted_pca,
            features=features,
            target=target,
            comment=comment,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            show_index=show_index,
            x_range=x_range,
            y_range=y_range
        )
        return x_range, y_range

    @classmethod
    def _check_batch_plot_parameters(
            cls,
            num_features,
            num_batches,
            batch,
            features,
            target,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            x_range,
            y_range
    ):
        """Check that parameters for a batch plot are valid.

        :param num_features: positive integer; expected number of features
        :param num_batches: positive integer; number of batches in the model
        :param batch: see docstring of plot_batch_map() for details
        :param features: see docstring of plot_batch_map() for details
        :param target: see docstring of plot_batch_map() for details
        :param highlight: see docstring of plot_batch_map() for details
        :param highlight_name: see docstring of plot_batch_map() for details
        :param reference: see docstring of plot_batch_map() for details
        :param explain_features: see docstring of plot_batch_map() for details
        :param explain_target: see docstring of plot_batch_map() for details
        :param x_range: see docstring of plot_batch_map() for details
        :param y_range: see docstring of plot_batch_map() for details
        :return: six return values: features, target, highlight, reference, explain_features, and explain_target as
            proper numpy arrays
        """
        features, target, highlight, reference, explain_features, explain_target = cls._check_plot_parameters(
            num_features=num_features,
            features=features,
            target=target,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            x_range=x_range,
            y_range=y_range
        )
        if batch < 1:
            raise ValueError("Parameter batch must be a positive integer.")
        if batch > num_batches:
            raise ValueError("Parameter batch may not exceed the number of batches in the model.")
        return features, target, highlight, reference, explain_features, explain_target

    @staticmethod
    def _get_batch_info(model, batch, explain_features):
        """Extract information for one batch from proset model.

        :param model: fitted proset model
        :param batch: see docstring of plot_batch_map() for details
        :param explain_features: see docstring of plot_batch_map() for details
        :return: dict with the following fields
        - index: 1D numpy array of non-negative integers; sample index for each prototype
        - prototypes: 2D numpy float array; prototype features for the selected batch reduced to active features
        - prototype_weights: 1D numpy array of positive floats; prototype weights
        - plot_weights: 1D numpy array of positive floats; weights for plotting prototypes
        - target: 1D numpy array; prototype target values
        - feature_weights 1D numpy array of positive floats; feature weights reduced to active features
        - feature_index: 1D numpy array of non-negative integers; indices of active features w.r.t. the original
          training samples
        """
        if explain_features is None:
            report = model.export(n_iter=batch)
        else:
            report = model.explain(X=explain_features, n_iter=batch)
        report = report[report["batch"] == batch]
        if report.shape[0] == 0:
            raise RuntimeError("The selected batch contains no prototypes, no plot can be generated.")
        feature_ix = np.array([i for i in range(report.shape[1]) if report.columns[i].endswith(" value")])
        keep_features = np.array([i for i in range(feature_ix.shape[0]) if not pd.isna(report.iloc[0, feature_ix[i]])])
        if len(keep_features) == 0:
            raise RuntimeError("The selected batch contains no features, no plot can be generated.")
        feature_ix = feature_ix[keep_features]
        prototypes = report.iloc[:, feature_ix].to_numpy(copy=True)
        prototype_weights = report["prototype weight"].to_numpy(copy=True)
        if explain_features is None:
            plot_weights = prototype_weights.copy()
        else:
            plot_weights = report["impact"].to_numpy(copy=True)
        feature_weights = report.iloc[0, feature_ix - 1].to_numpy(copy=True)
        feature_index = model.set_manager_.get_feature_weights()["feature_index"][keep_features]
        return {
            "index": report["sample"].to_numpy(copy=True),
            "prototypes": prototypes,
            "prototype_weights": prototype_weights,
            "plot_weights": plot_weights,
            "target": report["target"].to_numpy(copy=True),
            "feature_weights": feature_weights,
            "feature_index": feature_index
        }

    @staticmethod
    def _perform_weighted_pca(batch_info, random_state):
        """Perform weighted PCA on prototypes.

        :param batch_info: as return value of _get_batch_info()
        :param random_state: an instance of np.random.RandomState
        :return: dict with the following fields:
            - weighted_mean: mean value of prototype features weighted with prototype weight; new samples need to be
              centered with this mean and scaled with proset feature weights before applying PCA
            - pca: a fitted instance of sklearn.decomposition.PCA()
            - prototype_scores: 2D numpy float array; the first two components of weighted PCA applied to prototype
              features
        """
        weighted_mean = np.squeeze(
            np.sum(batch_info["prototypes"].transpose() * batch_info["prototype_weights"], axis=1)
            / np.sum(batch_info["prototype_weights"])
        )  # transpose() is used since multiplication broadcast over columns but weights apply to rows
        prototypes = (batch_info["prototypes"] - weighted_mean) * batch_info["feature_weights"]
        # use feature weights from batch for scaling, not weighted variance
        pca = PCA(n_components=2, random_state=random_state)
        pca.fit((prototypes.transpose() * np.sqrt(batch_info["prototype_weights"])).transpose())
        # use prototype weights on rows for sample-weighted pca
        return {
            "weighted_mean": weighted_mean,
            "pca": pca,
            "prototype_scores": pca.transform(prototypes)
        }

    @classmethod
    def _prepare_map_plot(cls, batch_info, weighted_pca, features, reference, explain_features, x_range, y_range):
        """Update features and plot ranges for batch map plot.

        :param batch_info: as return value of _get_batch_info()
        :param weighted_pca: as return value of _perform_weighted_pca()
        :param features: see docstring of plot_batch_map() for details
        :param reference: see docstring of plot_batch_map() for details
        :param explain_features: see docstring of plot_batch_map() for details
        :param x_range: see docstring of plot_batch_map() for details
        :param y_range: see docstring of plot_batch_map() for details
        :return: updated version features, reference, explain_features, x_range, and y_range
        """
        if features is not None:
            features = cls._apply_weighted_pca(features=features, batch_info=batch_info, weighted_pca=weighted_pca)
        if reference is not None:
            reference = cls._apply_weighted_pca(features=reference, batch_info=batch_info, weighted_pca=weighted_pca)
        if explain_features is not None:
            explain_features = cls._apply_weighted_pca(
                features=explain_features,
                batch_info=batch_info,
                weighted_pca=weighted_pca
            )
        x_range = cls._determine_plot_range(
            x_range,
            weighted_pca["prototype_scores"][:, 0],
            features[:, 0] if features is not None else None,
            reference[:, 0] if reference is not None else None,
            explain_features[:, 0] if explain_features is not None else None
        )
        y_range = cls._determine_plot_range(
            y_range,
            weighted_pca["prototype_scores"][:, 1],
            features[:, 1] if features is not None else None,
            reference[:, 1] if reference is not None else None,
            explain_features[:, 1] if explain_features is not None else None
        )
        return features, reference, explain_features, x_range, y_range

    @staticmethod
    def _apply_weighted_pca(batch_info, weighted_pca, features):
        """Apply weighted PCA to feature matrix.

        :param batch_info: dict; as return value of _get_batch_info()
        :param weighted_pca: dict; as return value of _perform_weighted_pca()
        :param features: 2D numpy float array; feature matrix
        :return: 2D numpy float array; first two principal components of active features using weighted PCA
        """
        return weighted_pca["pca"].transform(
            (features[:, batch_info["feature_index"]] - weighted_pca["weighted_mean"]) * batch_info["feature_weights"]
        )

    @classmethod
    def _create_map_plot(
            cls,
            model,
            model_name,
            alpha,
            batch,
            batch_info,
            weighted_pca,
            features,
            target,
            comment,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            show_index,
            x_range,
            y_range
    ):
        """Create batch map scatter plot.

        :param model: a fitted proset model
        :param model_name: see docstring of __init__() for details
        :param alpha: see docstring of __init__() for details
        :param batch: see docstring of plot_batch_map() for details
        :param batch_info: as return value of _get_batch_info()
        :param weighted_pca: as return value of _perform_weighted_pca()
        :param features: 2D numpy float array with 2 columns or None; features of supplementary points transformed to
            batch map coordinates
        :param target: see docstring of plot_batch_map() for details
        :param comment: see docstring of plot_batch_map() for details
        :param highlight: see docstring of plot_batch_map() for details
        :param highlight_name: see docstring of plot_batch_map() for details
        :param reference: 2D numpy float array with 2 columns and 1 row or None; reference point transformed to batch
            map coordinates
        :param explain_features: 2D numpy float array with 2 columns and 1 row or None; features of point to be
            explained transformed to batch map coordinates
        :param explain_target: see docstring of plot_batch_map() for details
        :param show_index: see docstring of plot_batch_map() for details
        :param x_range: see docstring of plot_batch_map() for details; None not allowed
        :param y_range: see docstring of plot_batch_map() for details; None not allowed
        :return: no return value; figure created
        """
        marker_size = cls._get_marker_size(plot_weights=batch_info["plot_weights"], show_features=features is not None)
        plt.figure()
        legend, legend_text = cls._create_scatter_plot(
            model=model,
            features=weighted_pca["prototype_scores"],
            target=batch_info["target"],
            from_report=True,
            marker_size=marker_size,
            alpha=alpha
        )
        if features is not None:
            cls._create_scatter_plot(
                model=model,
                features=features,
                target=target,
                from_report=False,
                marker_size=np.ones(features.shape[0], dtype=int) * SMALL_MARKER_SIZE * 2.0,
                alpha=alpha
            )
            if highlight is not None:
                legend.append(
                    cls._plot_highlights(features=features, highlight=highlight, marker_size=MEDIUM_MARKER_SIZE)
                )
                legend_text.append(highlight_name)
        if reference is not None:
            legend.append(cls._plot_single_point(features=reference, color="none"))
            legend_text.append("reference")
        if explain_features is not None:
            legend.append(cls._plot_single_point(
                features=explain_features,
                color=cls._get_map_target_color(model=model, target=explain_target)[0])
            )
            legend_text.append("to explain")
        if show_index:
            cls._print_index(
                features=weighted_pca["prototype_scores"],
                index=batch_info["index"],
                x_range=x_range,
                y_range=y_range
            )
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.grid("on")
        plt.xlabel("First PC")
        plt.ylabel("Second PC")
        if len(legend) > 0:
            plt.legend(legend, legend_text)
        title = "Batch {} map for {}".format(batch, model_name)
        if comment is not None:
            title += " ({})".format(comment)
        plt.suptitle(title)
        if features is None:
            if explain_features is None:
                plt.title("Marker area proportional to prototype weight")
            else:
                plt.title("Marker area proportional to prototype impact")
        else:
            plt.title("Large markers = prototypes / small markers = additional samples")

    @staticmethod
    def _get_marker_size(plot_weights, show_features):
        """Determine marker size for prototype scatter plot.

        :param plot_weights: 1D numpy array of positive floats; plot weights for prototypes
        :param show_features: boolean; whether the plot contains supplementary points
        :return: 1D numpy array of positive floats; squared marker size for matplotlib scatter plot
        """
        base_scale = LARGE_MARKER_SIZE ** 2.0  # plt.scatter() uses squared size as marker scale
        if not show_features:
            base_scale /= np.max(plot_weights)
            return base_scale * plot_weights
        return base_scale * np.ones(plot_weights.shape[0], dtype=float)

    @classmethod
    @abstractmethod
    def _create_scatter_plot(cls, model, features, target, from_report, marker_size, alpha):
        """Create scatter plot.

        :param model: a fitted proset model
        :param features: 2D numpy float array with two columns
        :param target: see docstring of plot_batch_map() for details
        :param from_report: boolean; whether the target value is taken from a model report or function input
        :param marker_size: 1D numpy array of positive floats; marker size
        :param alpha: see docstring of __init__() for details
        :return: two return values:
            - list of plot handles for creating a matplotlib legend
            - list of strings; legend text
        """
        NotImplementedError("Abstract method ModelPlots._create_scatter_plot() has no default implementation.")

    @classmethod
    @abstractmethod
    def _get_map_target_color(cls, model, target):
        """Determine target plot color for the map plot.

        :param model: see docstring of __init__() for details
        :param target: see docstring of plot_batch_map() for details
        :return: list of marker face colors for matplotlib.plot()
        """
        raise NotImplementedError("Abstract method ModelPlots._get_map_target_color() has no default implementation.")

    @staticmethod
    def _print_index(features, index, x_range, y_range):
        """Add index number to plot points.

        :param features: features: 2D numpy float array with 2 columns
        :param index: 1D numpy array of non-negative integers, may be encoded as floats; index values to print
        :param x_range: see docstring of plot_batch_map() for details
        :param y_range: see docstring of plot_batch_map() for details
        :return:
        """
        x_offset = (x_range[1] - x_range[0]) * TEXT_OFFSET
        y_offset = (y_range[1] - y_range[0]) * TEXT_OFFSET
        for i, value in enumerate(index):
            plt.text(x=features[i, 0] + x_offset, y=features[i, 1] + y_offset, s=str(int(value)))

    def plot_features(
            self,
            batch,
            features=None,
            target=None,
            comment=None,
            highlight=None,
            highlight_name=None,
            reference=None,
            explain_features=None,
            explain_target=None,
            show_index=True,
            make_single_figure=True,
            add_jitter=False
    ):
        """Create density plots and bivariate scatter plots for all features or combinations.

        :param batch: positive integer; batch to use
        :param features: 2D numpy float array or None; feature matrix of supplementary points to be plotted; sparse
            matrices or infinite/missing values not supported
        :param target: list-like object or None; target for supervised learning corresponding to features; must be None
            if features is None
        :param comment: string or None; a string will be added after the supertitle in brackets
        :param highlight: 1D numpy boolean array or None; indicator vector of supplementary points to highlight; this
            can only be not None if features is not None
        :param highlight_name: string or None; legend label for highlighted points; this must be provided iff highlight
            is not None
        :param reference: 2D numpy float array with one row or None; feature values for reference point; pass None to
            plot no reference point
        :param explain_features: 2D numpy float array with one row or None; feature values for point to be explained;
            pass None to plot no reference point
        :param explain_target: single value or None; target for point to be explained; this must be provided iff
            explain_features is not None
        :param show_index: boolean; whether to show the corresponding sampled index next to each prototype in the plot
        :param make_single_figure: boolean; whether to create a single figure with a grid of plots or multiple separate
            plots
        :param add_jitter: boolean or 1D numpy boolean array; whether to add random jitter to feature values; if passing
            an array, it must have one element per feature used to train the model
        :return: no return value; figure(s) generated
        """
        features, target, highlight, reference, explain_features, explain_target, add_jitter = \
            self._check_feature_plot_parameters(
                num_features=self._model.n_features_in_,
                num_batches=self._model.set_manager_.num_batches,
                batch=batch,
                features=features,
                target=target,
                highlight=highlight,
                highlight_name=highlight_name,
                reference=reference,
                explain_features=explain_features,
                explain_target=explain_target,
                add_jitter=add_jitter
            )
        batch_info = self._get_batch_info(model=self._model, batch=batch, explain_features=explain_features)
        feature_names, prototypes, features, reference, explain_features, bandwidths, axes = self._prepare_feature_plot(
            feature_names=self._feature_names,
            scale=self._scale,
            offset=self._offset,
            jitter_std=self._jitter_std,
            batch_info=batch_info,
            features=features,
            reference=reference,
            explain_features=explain_features,
            add_jitter=add_jitter,
            random_state=self._random_state
        )
        self._create_feature_plot(
            model=self._model,
            model_name=self._model_name,
            feature_names=feature_names,
            alpha=self._alpha,
            batch=batch,
            prototypes=prototypes,
            prototype_weights=batch_info["prototype_weights"],
            plot_weights=batch_info["plot_weights"],
            prototype_target=batch_info["target"],
            prototype_index=batch_info["index"] if show_index else None,
            bandwidths=bandwidths,
            features=features,
            target=target,
            comment=comment,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            make_single_figure=make_single_figure,
            axes=axes
        )

    @classmethod
    def _check_feature_plot_parameters(
            cls,
            num_features,
            num_batches,
            batch,
            features,
            target,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            add_jitter
    ):
        """Check that feature plot parameters are consistent.

        :param num_features: positive integer; expected number of features
        :param num_batches: positive integer; number of batches in the model
        :param batch: see docstring of plot_features() for details
        :param features: see docstring of plot_features() for details
        :param target: see docstring of plot_features() for details
        :param highlight: see docstring of plot_features() for details
        :param highlight_name: see docstring of plot_features() for details
        :param reference: see docstring of plot_features() for details
        :param explain_features: see docstring of plot_features() for details
        :param explain_target: see docstring of plot_features() for details
        :param add_jitter: see docstring of plot_features() for details
        :return: seven return values: features, target, highlight, reference, explain_features, explain_target, and
            add_jitter as proper numpy arrays
        """
        features, target, highlight, reference, explain_features, explain_target = cls._check_batch_plot_parameters(
            num_features=num_features,
            num_batches=num_batches,
            batch=batch,
            features=features,
            target=target,
            highlight=highlight,
            highlight_name=highlight_name,
            reference=reference,
            explain_features=explain_features,
            explain_target=explain_target,
            x_range=None,
            y_range=None
        )
        if isinstance(add_jitter, (bool, np.bool)):
            add_jitter = (np.zeros(num_features) * add_jitter).astype(bool)
        else:
            if len(add_jitter.shape) != 1:
                raise ValueError("Parameter add_jitter must be a boolean value or 1D array.")
            if add_jitter.shape[0] != num_features:
                raise ValueError(ERROR_MESSAGE_FEATURES.format("add_jitter", num_features))
            if add_jitter.dtype not in [bool, np.bool]:
                raise ValueError("Parameter add_jitter must be of boolean type.")
        return features, target, highlight, reference, explain_features, explain_target, add_jitter

    @classmethod
    def _prepare_feature_plot(
            cls,
            feature_names,
            scale,
            offset,
            jitter_std,
            batch_info,
            features,
            reference,
            explain_features,
            add_jitter,
            random_state
    ):
        """Prepare feature matrices for plotting.

        :param feature_names: list of strings; feature names
        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param jitter_std: see docstring of __init__() for details
        :param batch_info: as return value of get_batch_info()
        :param features: see docstring of plot_features() for details
        :param reference: see docstring of plot_features() for details
        :param explain_features: see docstring of plot_features() for details
        :param add_jitter: see docstring of plot_features() for details; must be an array
        :param random_state: an instance of np.random.RandomState
        :return: six return values:
            - feature names reduced to active features
            - batch_info["prototypes"], features, reference, and explain_features after applying scaling, offset, and
              jitter (no jitter applied to reference and explain_features); all reduced to active features for the
              current batch
            - batch_info["feature_weights"] after applying scaling
            - list of 1D numpy float arrays with two elements each; plot axis ranges for each feature dimension
        """
        feature_names = [feature_names[i] for i in batch_info["feature_index"]]
        scale = scale[batch_info["feature_index"]]
        offset = offset[batch_info["feature_index"]]
        add_jitter = add_jitter[batch_info["feature_index"]]
        prototypes = cls._apply_transforms(
            scale=scale,
            offset=offset,
            jitter_std=jitter_std,
            features=batch_info["prototypes"],
            add_jitter=add_jitter,
            random_state=random_state
        )
        if features is not None:
            features = cls._apply_transforms(
                scale=scale,
                offset=offset,
                jitter_std=jitter_std,
                features=features[:, batch_info["feature_index"]],
                add_jitter=add_jitter,
                random_state=random_state
            )
        if reference is not None:
            reference = cls._apply_transforms(
                scale=scale,
                offset=offset,
                jitter_std=jitter_std,
                features=reference[:, batch_info["feature_index"]],
                add_jitter=np.zeros_like(add_jitter),  # no jitter added to single point
                random_state=random_state
            )
        if explain_features is not None:
            explain_features = cls._apply_transforms(
                scale=scale,
                offset=offset,
                jitter_std=jitter_std,
                features=explain_features[:, batch_info["feature_index"]],
                add_jitter=np.zeros_like(add_jitter),  # no jitter added to single point
                random_state=random_state
            )
        axes = [cls._determine_plot_range(
            None,
            prototypes[:, ix],
            features[:, ix] if features is not None else None,
            reference[:, ix] if reference is not None else None,
            explain_features[:, ix] if explain_features is not None else None
        ) for ix in range(prototypes.shape[1])]
        return feature_names, prototypes, features, reference, explain_features, \
            scale / batch_info["feature_weights"], axes

    @staticmethod
    def _apply_transforms(scale, offset, jitter_std, features, add_jitter, random_state):
        """Apply scaling, offset, and jitter to a feature matrix.

        :param scale: see docstring of __init__() for details; None not allowed
        :param offset: see docstring of __init__() for details; None not allowed
        :param jitter_std: see docstring of __init__() for details
        :param features: 2D numpy float array
        :param add_jitter: see docstring of plot_features() for details; must be an array
        :param random_state: an instance of np.random.RandomState
        :return: 2D numpy float array; features after processing
        """
        features = features * scale + offset
        if np.any(add_jitter):
            features[:, add_jitter] += random_state.normal(
                scale=jitter_std,
                size=(features.shape[0], np.count_nonzero(add_jitter))
            )
        return features

    @classmethod
    def _create_feature_plot(
            cls,
            model,
            model_name,
            feature_names,
            alpha,
            batch,
            prototypes,
            prototype_weights,
            plot_weights,
            prototype_target,
            prototype_index,
            bandwidths,
            features,
            target,
            comment,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            make_single_figure,
            axes
    ):
        """Create feature plots.

        :param model: fitted proset model
        :param model_name: see docstring of __init__() for details
        :param feature_names: see docstring of __init__() for details
        :param alpha: see docstring of __init__() for details
        :param batch: see docstring of __init__() for details
        :param prototypes: as first return value of _prepare_feature_plot()
        :param prototype_weights: 1D numpy array of positive floats; prototype weights
        :param plot_weights: 1D numpy array of positive floats; prototype weights for plotting
        :param prototype_target: 1D numpy array; target values for prototypes as taken from the model report (already
            converted to numeric in case of classification)
        :param prototype_index: 1D numpy array of non-negative integers or None; sample indices for prototypes to be
            shown in the plot; pass None to show no indices
        :param bandwidths: as fifth return value of _prepare_feature_plot()
        :param features: see docstring of plot_features() for details; after processing with _prepare_feature_plot()
        :param target: see docstring of plot_features() for details
        :param comment: see docstring of plot_features() for details
        :param highlight: see docstring of plot_features() for details
        :param highlight_name: see docstring of plot_features() for details
        :param reference: see docstring of plot_features() for details; after processing with _prepare_feature_plot()
        :param explain_features: see docstring of plot_features() for details; after processing with
            _prepare_feature_plot()
        :param explain_target: see docstring of plot_features() for details
        :param make_single_figure: see docstring of plot_features() for details
        :param axes: as last return value of _prepare_feature_plot()
        :return: no return value; figure(s) generated
        """
        num_features = prototypes.shape[1]
        fid = plt.figure() if make_single_figure else None
        for row in range(num_features):
            for column in range(num_features):
                if make_single_figure or row >= column:  # suppress redundant plots if generating multiple figures
                    if make_single_figure:
                        plt.subplot(num_features, num_features, 1 + num_features * row + column)
                        if row == num_features - 1:
                            x_label = feature_names[column]
                        else:
                            x_label = None
                        if column == 0:
                            y_label = feature_names[row]
                            show_y_axis = "left"
                        elif column == num_features - 1:
                            y_label = feature_names[row]
                            show_y_axis = "right"
                        else:
                            y_label = None
                            show_y_axis = "no"
                    else:
                        plt.figure()
                        x_label = feature_names[column]
                        y_label = feature_names[row]
                        show_y_axis = "left"
                    if row == column:
                        supertitle, title = cls._get_titles(
                            batch=batch,
                            model_name=model_name,
                            comment=comment,
                            is_density=True,
                            make_single_figure=make_single_figure,
                            has_features=features is not None
                        )
                        cls._plot_feature_density(
                            model=model,
                            alpha=alpha,
                            prototypes=prototypes[:, [row]],
                            prototype_weights=prototype_weights,
                            prototype_target=prototype_target,
                            prototype_index=prototype_index,
                            bandwidth=bandwidths[row],
                            feature=features[:, [row]] if features is not None else None,
                            target=target,
                            highlight=highlight,
                            highlight_name=highlight_name,
                            reference=reference[:, [row]] if reference is not None else None,
                            explain_feature=explain_features[:, [row]] if explain_features is not None else None,
                            explain_target=explain_target,
                            x_label=x_label,
                            show_y_axis=show_y_axis,
                            show_legend=not make_single_figure or row == 0,
                            supertitle=supertitle,
                            title=title,
                            x_range=axes[row]
                        )
                    else:
                        supertitle, title = cls._get_titles(
                            batch=batch,
                            model_name=model_name,
                            comment=comment,
                            is_density=False,
                            make_single_figure=make_single_figure,
                            has_features=features is not None
                        )
                        cls._plot_points(
                            model=model,
                            alpha=alpha,
                            prototypes=prototypes[:, [column, row]],
                            plot_weights=plot_weights,
                            prototype_target=prototype_target,
                            prototype_index=prototype_index,
                            features=features[:, [column, row]] if features is not None else None,
                            target=target,
                            highlight=highlight,
                            highlight_name=highlight_name,
                            reference=reference[:, [column, row]] if reference is not None else None,
                            explain_features=explain_features[:, [column, row]]
                            if explain_features is not None else None,
                            explain_target=explain_target,
                            x_label=x_label,
                            y_label=y_label,
                            y_label_position="left" if show_y_axis == "left" else "right",
                            show_legend=not make_single_figure,
                            supertitle=supertitle,
                            title=title,
                            x_range=axes[column],
                            y_range=axes[row]
                        )
        if make_single_figure:
            fid.subplots_adjust(wspace=0, hspace=0)
            supertitle = "Batch {} feature plots for {}".format(batch, model_name)
            if comment is not None:
                supertitle += " ({})".format(comment)
            plt.suptitle(supertitle)

    @staticmethod
    def _get_titles(batch, model_name, comment, is_density, make_single_figure, has_features):
        if make_single_figure:
            return None, None
        if is_density:
            supertitle = "Batch {} distribution plot for {}".format(batch, model_name)
            if comment is not None:
                supertitle += " ({})".format(comment)
            title = "Solid curve & vertical lines = prototypes"
            if has_features:
                title += " / dashed curve & dots = additional samples"
            return supertitle, title
        supertitle = "Batch {} scatter plot for {}".format(batch, model_name)
        if comment is not None:
            supertitle += " ({})".format(comment)
        if has_features:
            title = "Large markers = prototypes / small markers = additional samples"
        else:
            title = "Marker area proportional to prototype weight"
        return supertitle, title

    @classmethod
    def _plot_feature_density(
            cls,
            model,
            alpha,
            prototypes,
            prototype_weights,
            prototype_target,
            prototype_index,
            bandwidth,
            feature,
            target,
            highlight,
            highlight_name,
            reference,
            explain_feature,
            explain_target,
            x_label,
            show_y_axis,
            show_legend,
            supertitle,
            title,
            x_range
    ):
        """Plot feature density and supplementary data for one feature.

        :param model: fitted proset model
        :param alpha: see docstring of __init__() for details
        :param prototypes: 2D numpy float array with a single column; prototype values along one dimension
        :param prototype_weights: 1D numpy array of positive floats; prototype weights
        :param bandwidth: positive float; bandwidth for kernel density estimator
        :param feature: 2D numpy float array with a single column or None; values for supplementary samples along one
            dimension
        :param target: see docstring of plot_features() for details
        :param highlight: see docstring of plot_features() for details
        :param highlight_name: see docstring of plot_features() for details
        :param reference: 2D numpy float array with a single row and column; value for reference point along one
            dimension
        :param explain_feature: 2D numpy float array with a single row and column; value for point to be explained along
            one dimension
        :param explain_target: see docstring of plot_features() for details
        :param x_label: string or None; x-axis label; pass None to suppress ticks and label
        :param show_y_axis: string; "left", "right", or "no" to position or hide y-axis ticks and label
        :param show_legend: boolean; whether to show the legend
        :param supertitle: string or None; plot supertitle; pass None to suppress supertitle
        :param title string or None; plot title; pass None to suppress title
        :param x_range: 1D numpy float array with two numbers; plot range for x-axis
        :return: no return value; plot generated
        """
        legend, legend_text, y_range, reference_level = cls._create_density_plot(
            model=model,
            alpha=alpha,
            feature=prototypes,
            sample_weights=prototype_weights,
            target=prototype_target,
            highlight=None,
            highlight_name=None,
            is_prototypes=True,
            index=prototype_index,
            bandwidth=bandwidth,
            x_range=x_range,
            y_max=0.0
        )
        if feature is not None:
            new_legend, new_legend_text, y_range, reference_level = cls._create_density_plot(
                model=model,
                alpha=alpha,
                feature=feature,
                sample_weights=None,
                target=target,
                highlight=highlight,
                highlight_name=highlight_name,
                is_prototypes=False,
                index=None,
                bandwidth=bandwidth,
                x_range=x_range,
                y_max=y_range[1]
            )
            if highlight is not None:  # add only highlight to legend
                legend.append(new_legend[-1])
                legend_text.append(new_legend_text[-1])
        if reference is not None:
            legend.append(cls._plot_single_point(
                features=np.reshape(np.hstack([reference[0], reference_level]), (1, 2)),
                color="none")
            )
            legend_text.append("reference")
        if explain_feature is not None:
            legend.append(cls._plot_single_point(
                features=np.reshape(np.hstack([explain_feature[0], reference_level]), (1, 2)),
                color=cls._get_map_target_color(model=model, target=explain_target)[0]
            ))
            legend_text.append("to explain")
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.grid("on")
        cls._add_labels(
            x_label=x_label,
            y_label="kernel density" if show_y_axis != "no" else None,
            y_label_position="left" if show_y_axis == "left" else "right",
            legend=legend if show_legend else None,
            legend_text=legend_text,
            supertitle=supertitle,
            title=title
        )

    @classmethod
    @abstractmethod
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

        :param model: fitted proset model
        :param alpha: see docstring of __init__() for details
        :param feature: 2D numpy float array with a single column; feature values along one dimension
        :param sample_weights: 1D numpy array of positive floats or None; sample weights for features; pass None for
            equal weights
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
        :param y_max: non-negative float; y-axis upper bound from previous call; pass 0.0 on first call
        :return: four return values:
            - list of matplotlib plot handles; plots to create legend for
            - list of strings; corresponding legend text
            - 1D numpy float array with two numbers; plot range for y-axis
            - float; y-axis value for plotting reference point and point to be explained
        """
        raise NotImplementedError("Abstract method ModelPlots._create_density_plot() has no default implementation.")

    @staticmethod
    def _add_labels(x_label, y_label, y_label_position, legend, legend_text, supertitle, title):
        """Add labels and legends based on plot settings.

        :param x_label: string or None; x-axis label; pass None to suppress ticks and label
        :param y_label: string or None; y-axis label; pass None to suppress ticks and label
        :param y_label_position: string; 'left' or 'right'; position of y-axis ticks and labels; only used if y_label is
            not None
        :param legend: list of matplotlib plot handles or None; annotate these plots; pass None to suppress the legend
        :param legend_text: list of strings; names for annotated plots; only used if legend is not None
        :param supertitle: string or None; plot supertitle; pass None to suppress supertitle
        :param title: string or None; plot title; pass None to suppress title
        :return:
        """
        ax = plt.gca()
        if x_label is None:
            ax.set_xticklabels([])
        else:
            plt.xlabel(x_label)
        if y_label is None:
            ax.set_yticklabels([])
        else:
            ax.yaxis.set_ticks_position(y_label_position)
            ax.yaxis.set_label_position(y_label_position)
            plt.ylabel(y_label)
        if legend is not None:
            plt.legend(legend, legend_text)
        if supertitle is not None:
            plt.suptitle(supertitle)
        if title is not None:
            plt.title(title)

    @classmethod
    def _plot_points(
            cls,
            model,
            alpha,
            prototypes,
            plot_weights,
            prototype_target,
            prototype_index,
            features,
            target,
            highlight,
            highlight_name,
            reference,
            explain_features,
            explain_target,
            x_label,
            y_label,
            y_label_position,
            show_legend,
            supertitle,
            title,
            x_range,
            y_range
    ):
        """Create scatter plot with supplementary data for one pair of features.

        :param model: fitted proset model
        :param alpha: see docstring of __init__() for details
        :param prototypes: 2D numpy float array with a two columns; prototype values along two dimensions
        :param plot_weights: 1D numpy array of positive floats; prototype weights for plotting
        :param prototype_target: 1D numpy array; target values for prototypes as taken from the model report (already
            converted to numeric in case of classification)
        :param prototype_index: 1D numpy array of non-negative integers or None; sample indices for prototypes to be
            shown in the plot; pass None to show no indices
        :param features: 2D numpy float array with a two columns or None; values for supplementary samples along two
            dimensions
        :param target: see docstring of plot_features() for details
        :param highlight: see docstring of plot_features() for details
        :param highlight_name: see docstring of plot_features() for details
        :param reference: 2D numpy float array with a single row and two columns; value for reference point along one
            dimension
        :param explain_features: 2D numpy float array with a single row and two columns; value for point to be explained
            along one dimension
        :param explain_target: see docstring of plot_features() for details
        :param x_label: string or None; x-axis label; pass None to suppress ticks and label
        :param y_label: string or None; y-axis label; pass None to suppress ticks and label
        :param y_label_position: string; "left" or "right"; only used if y_label is not None
        :param show_legend: boolean; whether to show the legend
        :param supertitle: string or None; plot supertitle; pass None to suppress supertitle
        :param title string or None; plot title; pass None to suppress title
        :param x_range: 1D numpy float array with two numbers; plot range for x-axis
        :param y_range: 1D numpy float array with two numbers; plot range for y-axis
        :return: no return value; plot generated
        """
        marker_size = cls._get_marker_size(plot_weights=plot_weights, show_features=features is not None)
        legend, legend_text = cls._create_scatter_plot(
            model=model,
            features=prototypes,
            target=prototype_target,
            from_report=True,
            marker_size=marker_size,
            alpha=alpha
        )
        if features is not None:
            cls._create_scatter_plot(
                model=model,
                features=features,
                target=target,
                from_report=False,
                marker_size=np.ones(features.shape[0], dtype=int) * SMALL_MARKER_SIZE * 2.0,
                alpha=alpha
            )
            if highlight is not None:
                legend.append(
                    cls._plot_highlights(features=features, highlight=highlight, marker_size=MEDIUM_MARKER_SIZE)
                )
                legend_text.append(highlight_name)
        if reference is not None:
            legend.append(cls._plot_single_point(features=reference, color="none"))
            legend_text.append("reference")
        if explain_features is not None:
            legend.append(cls._plot_single_point(
                features=explain_features,
                color=cls._get_map_target_color(model=model, target=explain_target)[0])
            )
            legend_text.append("to explain")
        if prototype_index is not None:
            cls._print_index(
                features=prototypes,
                index=prototype_index,
                x_range=x_range,
                y_range=y_range
            )
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.grid("on")
        cls._add_labels(
            x_label=x_label,
            y_label=y_label,
            y_label_position=y_label_position,
            legend=legend if show_legend else None,
            legend_text=legend_text,
            supertitle=supertitle,
            title=title
        )
