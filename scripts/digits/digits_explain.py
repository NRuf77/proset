"""Explain proset classifier trained on the UCI ML hand-written digits dataset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from copy import deepcopy
import gzip
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import proset.utility as utility


# noinspection PyShadowingNames
def plot_explanation(
        test_ix,
        true_class,
        true_proba,
        predicted_class,
        predicted_proba,
        test_features,
        test_features_raw,
        train_features,
        train_features_raw,
        active_features,
        model_explain,
        shap_explainer
):
    """Create explanatory plots for one sample of digits data.

    :param test_ix: integer; index of sample w.r.t. test data
    :param true_class: integer in [0, 9]; true class
    :param true_proba: float in [0.0, 1.9]; probability for true class assigned by the proset classifier
    :param predicted_class: integer in [0, 9]; class predicted by the proset classifier (majority class)
    :param predicted_proba: float in [0.0, 1.9]; probability for predicted class assigned by the proset classifier
    :param test_features: 2D numpy float array with one row and 64 columns; transformed features for test sample
    :param test_features_raw: 2D numpy float array with one row and 64 columns; original features for test sample
    :param train_features: 2D numpy float array with 64 columns; transformed features for training data
    :param train_features_raw: 2D numpy float array with 64 columns; original features for training data
    :param active_features: 1D numpy array of integers in [0, 64]; indices of active features
    :param model_explain: pandas data frame; proset classifier explanation report for the test sample
    :param shap_explainer: SHAP explainer wrapping the proset classifier
    :return: no return value; matplotlib figure generated
    """
    familiarity = float(re.search("familiarity ([0-9.]+)", model_explain["sample name"].iloc[0]).group(1))
    # parse quantile of familiarity from description of test sample
    model_explain = model_explain[np.logical_not(pd.isna(model_explain["batch"]))]
    # keep only data related to prototypes
    model_explain.reset_index(drop=True)
    plt.figure()
    shap_values = shap_explainer(test_features[:, active_features])
    _plot_column(
        column_ix=1,
        first_title="test sample",
        true_class=true_class,
        predicted_class=predicted_class,
        features=np.squeeze(test_features_raw),
        active_features=active_features,
        shap_values_truth=shap_values[0, :, true_class].values,
        shap_values_predicted=shap_values[0, :, predicted_class].values
    )
    for i in range(5):
        train_ix = int(model_explain["sample"].iloc[i])
        train_class = int(model_explain["target"].iloc[i])
        shap_values = shap_explainer(train_features[train_ix:(train_ix + 1), active_features])
        _plot_column(
            column_ix=i + 2,
            first_title="batch {}, sample {},\nclass '{}', impact {:.1f} %".format(
                int(model_explain["batch"].iloc[i]),
                train_ix,
                train_class,
                100.0 * model_explain["p class {}".format(train_class)].iloc[i]
            ),
            true_class=true_class,
            predicted_class=predicted_class,
            features=train_features_raw[train_ix, :],
            active_features=active_features,
            shap_values_truth=shap_values[0, :, true_class].values,
            shap_values_predicted=shap_values[0, :, predicted_class].values
        )
    supertitle = "\n".join([
        "Test sample '{}': true class '{}' ({:.1f} %), prediction '{}' ({:.1f} %), familiarity {:.2f}",
        "Columns 2 to 6 show five prototypes with largest impact on classification results"
    ])
    plt.suptitle(supertitle.format(
        test_ix, true_class, 100.0 * true_proba, predicted_class, 100.0 * predicted_proba, familiarity
    ))


# noinspection PyShadowingNames
def _plot_column(
        column_ix,
        first_title,
        true_class,
        predicted_class,
        features,
        active_features,
        shap_values_truth,
        shap_values_predicted
):
    """Generate one column of subplots for the explanatory plot.

    :param column_ix: integer in [1, 6]; column index for subplot
    :param first_title: string: title for first subplot in column
    :param true_class: see docstring of plot_explanation() for details
    :param predicted_class: see docstring of plot_explanation() for details
    :param features: 2D numpy float array with one row and 64 columns; original features for one sample
    :param active_features: see docstring of plot_explanation() for details
    :param shap_values_truth: 1D numpy float array; SHAP values for all active features with respect to the true class
    :param shap_values_predicted: 1D numpy float array; SHAP values for all active features with respect to the
        predicted class
    :return: no return value; plots generated in the current figure
    """
    plt.subplot(3, 6, column_ix)
    _plot_digit(features)
    _suppress_tick_labels()
    plt.title(first_title)
    plt.subplot(3, 6, column_ix + 6)
    _plot_digit(features)
    _plot_markers(positions=active_features, colors=_make_colors(shap_values_truth))
    plt.title("SHAP values for truth '{}'".format(true_class))
    _suppress_tick_labels()
    plt.subplot(3, 6, column_ix + 12)
    _plot_digit(features)
    _plot_markers(positions=active_features, colors=_make_colors(shap_values_predicted))
    plt.title("SHAP values for pred. '{}'".format(predicted_class))
    _suppress_tick_labels()


# noinspection PyShadowingNames
def _plot_digit(features):
    """Create an image plot of a single digit.

    :param features: 1D numpy float array of length 64; greyscale value from 0.0 to 16.0 for one digit
    :return: no return value; image plot created
    """
    image = np.zeros((8, 8, 3), dtype=float)
    for i in range(8):
        for j in range(8):
            image[i, j, :] = (16.0 - features[i * 8 + j]) / 16.0
    plt.imshow(X=image)


def _plot_markers(positions, colors):
    """Add colored markers to image plot.

    :param positions: 1D numpy of integers in [0, 63]; index of markers w.r.t. feature space
    :param colors: 2D numpy float array with one row per element of positions and three columns; marker RGB values
    :return: no return value; scatter plot created
    """
    x = np.array([position % 8 for position in positions])
    y = np.array([position // 8 for position in positions])
    plt.scatter(x, y, c=colors, marker="s", edgecolors="k")


def _make_colors(shap_values):
    """Convert shap values to colors.

    :param shap_values: 1D numpy float array; SHAP values for active features
    :return: 2D numpy float array with one row per element of shap_values and three columns; RGB values running from
        blue (large negative impact) over grey (neutral) to red (large positive impact); shap_values are scaled using
        maximum absolute value
    """
    scaling = np.max(np.abs(shap_values)) * 1.25
    return np.vstack([
        np.array([0.8, 0.8 - value / scaling, 0.8 - value / scaling]) if value >= 0.0 else
        np.array([0.8 + value / scaling, 0.8 + value / scaling, 0.8])
        for value in shap_values
    ])


def _suppress_tick_labels():
    """Suppress tick labels on the current matplotlib axes.

    :return: no return value; axes formatted
    """
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])


print("* Apply user settings")
input_path = "scripts/results"
input_files = [
    "digits_20b_model.gz",
    "digits_20b_beta_50_model.gz"
]
print("  Select input file:")
for i, file_name in enumerate(input_files):
    print("  {} - {}".format(i, file_name))
choice = int(input())
input_file = input_files[choice]
model_name = input_file.replace(".gz", "")

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Determine reference point")
scale = np.sqrt(result["model"]["transform"].var_)
scale[scale == 0.0] = 1.0  # some features have zero variance
offset = result["model"]["transform"].mean_
train_features = result["model"]["transform"].transform(result["data"]["X_train"])
train_labels = result["data"]["y_train"]
label_names = [str(i) for i in result["model"].classes_]
reference = utility.choose_reference_point(
    features=train_features,
    model=result["model"]["model"],
    scale=scale,
    offset=offset
)
utility.print_point_report(
    reference=reference,
    feature_names=result["data"]["feature_names"],
    target_names=label_names
)

print("* Show global results")
test_features = result["model"]["transform"].transform(result["data"]["X_test"])
test_labels = result["data"]["y_test"]
prediction, familiarity = result["model"]["model"].predict(X=test_features, compute_familiarity=True)
misclassified = prediction != test_labels
plotter = utility.ClassifierPlots(
    model=result["model"]["model"],
    model_name=model_name,
    feature_names=result["data"]["feature_names"],
    scale=scale,
    offset=offset
)
x_range, y_range = plotter.plot_batch_map(
    batch=1,
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    reference=reference["features_raw"],
    show_index=False
)
# no feature plots as there are too many features

print("* Create SHAP explainer")
shrunk_model = deepcopy(result["model"]["model"])
shrunk_model.shrink()
active_features = reference["active_features"]
active_feature_names = result["data"]["feature_names"][active_features]
explainer = shap.Explainer(
    model=shrunk_model.predict_proba,
    masker=reference["features_raw"][0:1, active_features],
    feature_names=active_feature_names
)

print("* Explain misclassified digits")
for test_ix in np.where(misclassified)[0]:
    test_features_ix = test_features[test_ix:(test_ix + 1), :]
    test_features_raw_ix = result["data"]["X_test"][test_ix:(test_ix + 1), :]
    proba = result["model"]["model"].predict_proba(test_features_ix)
    true_class = test_labels[test_ix]
    predicted_class = prediction[test_ix]
    explain = result["model"]["model"].explain(
        X=test_features_ix,
        y=true_class,
        familiarity=familiarity,
        sample_name="test sample {}".format(test_ix),
        feature_names=result["data"]["feature_names"],
        scale=scale,
        offset=offset
    )
    plot_explanation(
        test_ix=test_ix,
        true_class=true_class,
        true_proba=proba[0, true_class],
        predicted_class=predicted_class,
        predicted_proba=proba[0, predicted_class],
        test_features=test_features_ix,
        test_features_raw=test_features_raw_ix,
        train_features=train_features,
        train_features_raw=result["data"]["X_train"],
        active_features=active_features,
        model_explain=explain,
        shap_explainer=explainer
    )

print("* Done")
