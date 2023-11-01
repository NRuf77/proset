"""Score proset classifier trained on a deterministic checkerboard pattern.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss, roc_auc_score

import proset.utility as utility


print("* Apply user settings")
input_path = "scripts/results"
output_path = "scripts/reports"
input_files = [
    "checker_10b_model.gz",
    "checker_10b_beta_50_model.gz"
]
print("  Select input file:")
for i, file_name in enumerate(input_files):
    print("  {} - {}".format(i, file_name))
choice = int(input())
input_file = input_files[choice]
export_file = input_file.replace(".gz", "_export.xlsx")
model_name = input_file.replace(".gz", "")

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Generate prototype report")
scale = np.sqrt(result["model"]["transform"].var_)
offset = result["model"]["transform"].mean_
export = result["model"]["model"].export(feature_names=result["data"]["feature_names"], scale=scale, offset=offset)
utility.write_report(file_path=os.path.join(output_path, export_file), report=export)

print("* Show results")
train_features = result["model"]["transform"].transform(result["data"]["X_train"])
train_labels = result["data"]["y_train"]
test_features = result["model"]["transform"].transform(result["data"]["X_test"])
test_labels = result["data"]["y_test"]
prediction = result["model"]["model"].predict(test_features)
probabilities = result["model"]["model"].predict_proba(test_features)
active_features = result["model"]["model"].set_manager_.get_active_features()
misclassified = prediction != test_labels
prototype_ix = np.zeros(train_features.shape[0], dtype=bool)
prototype_ix[export["sample"][np.logical_not(pd.isna(export["batch"]))].drop_duplicates().to_numpy().astype(int)] = True
print("- Hyperparameter selection")
utility.print_hyperparameter_report(result)
print("-  Final model")
print("log-loss = {:.2f}".format(log_loss(y_true=test_labels, y_pred=probabilities)))
print("roc-auc  = {:.2f}".format(roc_auc_score(y_true=test_labels, y_score=probabilities[:, 1])))
print("number of active features = {}".format(active_features.shape[0]))
print("number of prototypes = {}\n".format(result["model"]["model"].set_manager_.get_num_prototypes()))
print("- Selected features and weights")
utility.print_feature_report(model=result["model"]["model"], feature_names=result["data"]["feature_names"])
print("- Classification report")
print(classification_report(y_true=test_labels, y_pred=prediction))
utility.plot_select_results(result=result, model_name=model_name)
plotter = utility.ClassifierPlots(
    model=result["model"]["model"],
    model_name=model_name,
    feature_names=result["data"]["feature_names"],
    scale=scale,
    offset=offset
)
x_range, y_range = plotter.plot_surface(
    features=train_features,
    target=train_labels,
    comment="training samples",
    highlight=prototype_ix,
    highlight_name="prototypes",
    use_proba=True
)
plotter.plot_surface(
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    x_range=x_range,
    y_range=y_range,
    use_proba=True
)
plotter.plot_surface(
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    x_range=x_range,
    y_range=y_range,
)
x_range, y_range = plotter.plot_batch_map(
    batch=1,
    features=train_features,
    target=train_labels,
    comment="training samples",
    show_index=False
)
plotter.plot_batch_map(
    batch=1,
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    x_range=x_range,
    y_range=y_range,
    show_index=False
)
plotter.plot_batch_map(batch=1, x_range=x_range, y_range=y_range, show_index=False)
plotter.plot_features(
    batch=1,
    features=test_features,
    target=test_labels,
    comment="test samples",
    highlight=misclassified,
    highlight_name="misclassified",
    show_index=False
)

print("* Done")
