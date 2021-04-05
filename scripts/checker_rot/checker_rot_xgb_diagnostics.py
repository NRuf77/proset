"""Score XGBoost classifier trained on the rotated checkerboard pattern.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, log_loss, roc_auc_score

import proset
from proset.benchmarks import print_xgb_classifier_report


print("* Apply user settings")
input_path = "scripts/results"
input_file = "checker_rot_xgb_model.gz"

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
truth = result["data"]["y_test"]
prediction = result["model"].predict(result["data"]["X_test"])
probabilities = result["model"].predict_proba(result["data"]["X_test"])

print("- Hyperparameter selection")
print_xgb_classifier_report(result)
print("-  Final model")
print("active features   = {}".format(np.sum(result["model"].feature_importances_ > 0.0)))
print("log-loss          = {:.2f}".format(log_loss(y_true=truth, y_pred=probabilities)))
print("roc-auc           = {:.2f}".format(roc_auc_score(y_true=truth, y_score=probabilities[:, 1])))
print("- Classification report")
print(classification_report(y_true=truth, y_pred=prediction))

proset.plot_decision_surface(
    features=result["data"]["X_test"],
    target=result["data"]["y_test"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="checker rot XGBoost classifier"
)
proset.plot_decision_surface(
    features=result["data"]["X_test"],
    target=result["data"]["y_test"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="checker rot XGBoost classifier",
    use_proba=True
)

print("* Done")
