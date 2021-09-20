"""Score CGBoost classifier trained on wine classification data.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, log_loss, roc_auc_score

from proset.benchmarks import print_xgb_classifier_report


print("* Apply user settings")
input_path = "scripts/results"
input_file = "wine_xgb_model.gz"

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
test_labels = result["data"]["y_test"]
prediction = result["encoder"].inverse_transform(result["model"].predict(result["data"]["X_test"]))
probabilities = result["model"].predict_proba(result["data"]["X_test"])
print("- Hyperparameter selection")
print_xgb_classifier_report(result)
print("-  Final model")
print("active features   = {}".format(np.sum(result["model"].feature_importances_ > 0.0)))
print("log-loss          = {:.2f}".format(log_loss(y_true=test_labels, y_pred=probabilities)))
print("roc-auc           = {:.2f}".format(roc_auc_score(y_true=test_labels, y_score=probabilities, multi_class="ovo")))
print("- Classification report")
print(classification_report(y_true=test_labels, y_pred=prediction))

print("* Done")
