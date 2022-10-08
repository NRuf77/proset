"""Explain proset classifier trained on breast cancer data.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from copy import deepcopy
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
input_file = "cancer_10b_model.gz"
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
    reference=reference["features_raw"]
)
plotter.plot_features(
    batch=1,
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    reference=reference["features_raw"],
    show_index=False
)

print("* Compute global SHAP values")
shrunk_model = deepcopy(result["model"]["model"])
shrunk_model.shrink()
active_features = reference["active_features"]
active_feature_names = result["data"]["feature_names"][active_features]
explainer = shap.Explainer(
    model=shrunk_model.predict_proba,
    masker=reference["features_raw"][0:1, active_features],
    feature_names=active_feature_names
)
shap_values = explainer(test_features[:, active_features])
plt.figure()
shap.plots.bar(shap_values[:, :, 0])
plt.title("Average SHAP values for binary prediction")

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
    explain_target=worst_label,
    show_index=False
)

print("* Compute SHAP values for single point")
explain = shap_values[worst_ix, :, 0]
shap.plots.force(
    base_value=explain.base_values,
    shap_values=explain.values,
    features=test_features[worst_ix:(worst_ix + 1), active_features],
    feature_names=active_feature_names,
    matplotlib=True
)
plt.gca().set_position([0.1, -0.25, 0.8, 0.8])  # force plot messes up the axes position within the figure
plt.suptitle("SHAP force plot: probability for class '{}' is {:0.2f}, true class is '{}'".format(
    result["model"].classes_[0], proba[worst_ix, 0], worst_label
))

print("* Show cross-sections of decision surface")
importance = np.mean(np.abs(shap_values[:, :, 0].values), axis=0)
top_two = active_features[np.argsort(importance)[-1:-3:-1]]
plotter.plot_surface(
    features=test_features,
    target=None,  # suppress sample plot, features only used to determine plot ranges
    baseline=worst_point["features_raw"],
    plot_index=top_two,
    comment="globally most important features",
    reference=reference["features_raw"],
    explain_features=worst_point["features_raw"],
    explain_target=worst_label,
    familiarity=familiarity,
    quantiles=(0.01, 0.05),
    use_proba=True
)
importance = np.abs(explain.values)
top_two = active_features[np.argsort(importance)[-1:-3:-1]]
plotter.plot_surface(
    features=test_features,
    target=None,  # suppress sample plot, features only used to determine plot ranges
    baseline=worst_point["features_raw"],
    plot_index=top_two,
    comment="most important features for single point",
    reference=reference["features_raw"],
    explain_features=worst_point["features_raw"],
    explain_target=worst_label,
    familiarity=familiarity,
    quantiles=(0.01, 0.05),
    use_proba=True
)

print("* Done")
