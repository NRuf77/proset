"""Score k-nearest neighbor classifier trained on two features of Fisher's iris data.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, log_loss, roc_auc_score

import proset.utility as utility


print("* Apply user settings")
input_path = "scripts/results"
input_file = "iris_2f_knn_model.gz"
model_name = input_file.replace(".gz", "")

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
test_features = result["model"]["transform"].transform(result["data"]["X_test"])
test_labels = result["data"]["y_test"]
prediction = result["model"]["model"].predict(test_features)
probabilities = result["model"]["model"].predict_proba(test_features)
print("- Hyperparameter selection")
print("optimal k         = {}".format(result["info"]["k_grid"][result["info"]["best_index"]]))
print("optimal log-loss  = {:.2f}".format(result["info"]["scores"][result["info"]["best_index"]]))
print("threshold         = {:.2f}".format(result["info"]["threshold"]))
print("selected k        = {}".format(result["info"]["k_grid"][result["info"]["selected_index"]]))
print("selected log-loss = {:.2f}".format(result["info"]["scores"][result["info"]["selected_index"]]))
print("-  Final model")
print("log-loss          = {:.2f}".format(log_loss(y_true=test_labels, y_pred=probabilities)))
print("roc-auc           = {:.2f}".format(roc_auc_score(y_true=test_labels, y_score=probabilities, multi_class="ovo")))
print("- Classification report")
print(classification_report(y_true=test_labels, y_pred=prediction))
scale = np.sqrt(result["model"]["transform"].var_)
offset = result["model"]["transform"].mean_
plotter = utility.ClassifierPlots(
    model=result["model"]["model"],
    model_name=model_name,
    feature_names=result["data"]["feature_names"],
    scale=scale,
    offset=offset
)
train_features = result["model"]["transform"].transform(result["data"]["X_train"])
train_labels = result["data"]["y_train"]
misclassified = prediction != test_labels
x_range, y_range = plotter.plot_surface(
    features=train_features,
    target=train_labels,
    comment="training samples",
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

print("* Done")
