"""Explain proset classifier trained on two features of Fisher's iris data.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap

import proset.utility as utility


print("* Apply user settings")
input_path = "scripts/results"
output_path = "scripts/reports"
input_files = [
    "iris_2f_10b_model.gz",
    "iris_2f_10b_beta_50_model.gz"
]
print("  Select input file:")
for i, file_name in enumerate(input_files):
    print("  {} - {}".format(i, file_name))
choice = int(input())
input_file = input_files[choice]
export_file = input_file.replace(".gz", "_explain.xlsx")
model_name = input_file.replace(".gz", "")

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Determine reference point")
scale = np.sqrt(result["model"]["transform"].var_)
offset = result["model"]["transform"].mean_
train_features = result["model"]["transform"].transform(result["data"]["X_train"])
train_labels = result["data"]["y_train"]
reference = utility.choose_reference_point(
    features=train_features,
    model=result["model"]["model"],
    scale=scale,
    offset=offset
)
utility.print_point_report(
    reference=reference,
    feature_names=result["data"]["feature_names"],
    target_names=result["model"].classes_
)

print("* Show global results")
test_features = result["model"]["transform"].transform(result["data"]["X_test"])
test_labels = result["data"]["y_test"]
prediction, familiarity = result["model"]["model"].predict(test_features, compute_familiarity=True)
active_features = result["model"]["model"].set_manager_.get_active_features()
misclassified = prediction != test_labels
plotter = utility.ClassifierPlots(
    model=result["model"]["model"],
    model_name=model_name,
    feature_names=result["data"]["feature_names"],
    scale=scale,
    offset=offset
)
plotter.plot_surface(
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    reference=reference["features_raw"],
    familiarity=familiarity,
    use_proba=True
)
if active_features.shape[0] == 2:
    x_range, y_range = plotter.plot_batch_map(
        batch=1,
        features=test_features,
        target=test_labels,
        comment="test samples",
        highlight=misclassified,
        highlight_name="misclassified",
        reference=reference["features_raw"]
    )
    plotter.plot_features(
        batch=1,
        features=test_features,
        target=test_labels,
        comment="test samples",
        highlight=misclassified,
        highlight_name="misclassified",
        reference=reference["features_raw"]
    )
else:
    print("! Cannot generate scatter plots for less than 2 active features\n")

print("* Compute global SHAP values")
explainer = shap.Explainer(
    model=result["model"]["model"].predict_proba,
    masker=reference["features_raw"][0:1, :] + 1e-3,
    # add offset as workaround for shap bug: trying to explain the reference point does not always work and iris data
    # has duplicate points
    feature_names=result["data"]["feature_names"]
)
shap_values = explainer(test_features)
for i, label in enumerate(result["model"].classes_):
    plt.figure()
    shap.plots.bar(shap_values[:, :, i])
    plt.title("Average SHAP values for class {} prediction".format(label))

print("* Find single point with worst classification result")
proba = result["model"]["model"].predict_proba(test_features)
truth_int = result["model"]["model"].label_encoder_.transform(test_labels)
worst_ix = np.argmin(proba[np.arange(test_labels.shape[0]), truth_int])
worst_features = test_features[worst_ix:(worst_ix + 1), :]
worst_label = test_labels[worst_ix]
worst_point = {
    "index": worst_ix,
    "features_raw": worst_features,
    "features_processed": worst_features[:, active_features] * scale[active_features] + offset[active_features],
    "prediction": proba[worst_ix, :],
    "num_features": test_features.shape[1],
    "active_features": active_features
}  # use active_features here to ensure same order of content as reference
print("  True class = '{}'".format(test_labels[worst_ix]))
utility.print_point_report(
    reference=worst_point,
    feature_names=result["data"]["feature_names"],
    target_names=result["model"].classes_
)

print("* Generate explanation report")
explain = result["model"]["model"].explain(
    X=worst_point["features_raw"],
    y=worst_label,
    familiarity=familiarity,
    sample_name="test sample {}".format(worst_ix),
    feature_names=result["data"]["feature_names"],
    scale=scale,
    offset=offset
)
utility.write_report(file_path=os.path.join(output_path, export_file), report=explain)

print("* Show results for single point")
plotter.plot_surface(
    features=train_features,
    target=train_labels,
    comment="training samples",
    reference=reference["features_raw"],
    explain_features=worst_point["features_raw"],
    explain_target=worst_label,
    familiarity=familiarity,
    use_proba=True
)
if active_features.shape[0] == 2:
    # noinspection PyUnboundLocalVariable
    plotter.plot_batch_map(
        batch=1,
        features=train_features,
        target=train_labels,
        comment="training samples",
        reference=reference["features_raw"],
        explain_features=worst_point["features_raw"],
        explain_target=worst_label,
        x_range=x_range,
        y_range=y_range
    )
    plotter.plot_batch_map(
        batch=1,
        reference=reference["features_raw"],
        explain_features=worst_point["features_raw"],
        explain_target=worst_label,
        x_range=x_range,
        y_range=y_range
    )
    plotter.plot_features(
        batch=1,
        features=train_features,
        target=train_labels,
        comment="training samples",
        reference=reference["features_raw"],
        explain_features=worst_point["features_raw"],
        explain_target=worst_label
    )

print("* Compute SHAP values for single point")
for i in range(proba.shape[1]):
    explain = shap_values[worst_ix, :, i]
    shap.plots.force(
        base_value=explain.base_values,
        shap_values=explain.values,
        features=worst_features,
        feature_names=result["data"]["feature_names"],
        matplotlib=True
    )
    plt.gca().set_position([0.1, -0.25, 0.8, 0.8])  # force plot messes up the axes position within the figure
    plt.suptitle("SHAP force plot: probability for class '{}' is {:0.2f}, true class is '{}'".format(
        result["model"].classes_[i], proba[worst_ix, i], worst_label
    ))

print("* Done")
