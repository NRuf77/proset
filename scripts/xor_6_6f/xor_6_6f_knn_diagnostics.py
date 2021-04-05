"""Score k-nearest neighbor classifier trained on the 'continuous XOR' problem with 6 relevant and 6 irrelevant
features.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, log_loss, roc_auc_score

import proset


print("* Apply user settings")
input_path = "scripts/results"
input_file = "xor_6_6f_knn_model.gz"

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
truth = result["data"]["y_test"]
prediction = result["model"].predict(result["data"]["X_test"])
probabilities = result["model"].predict_proba(result["data"]["X_test"])

print("- Hyperparameter selection")
print("optimal k         = {}".format(result["info"]["k_grid"][result["info"]["best_index"]]))
print("optimal log-loss  = {:.2f}".format(result["info"]["scores"][result["info"]["best_index"]]))
print("threshold         = {:.2f}".format(result["info"]["threshold"]))
print("selected k        = {}".format(result["info"]["k_grid"][result["info"]["selected_index"]]))
print("selected log-loss = {:.2f}".format(result["info"]["scores"][result["info"]["selected_index"]]))
print("-  Final model")
print("log-loss          = {:.2f}".format(log_loss(y_true=truth, y_pred=probabilities)))
print("roc-auc           = {:.2f}".format(roc_auc_score(y_true=truth, y_score=probabilities[:, 1])))
print("- Classification report")
print(classification_report(y_true=truth, y_pred=prediction))

ix = np.prod(result["data"]["X_test"][:, 2:6], axis=1) >= 0
# select test samples which have identical class based on the first two features
proset.plot_decision_surface(
    features=result["data"]["X_test"][ix, :2],
    target=result["data"]["y_test"][ix],
    model=result["model"],
    feature_names=result["data"]["feature_names"][:2],
    class_labels=result["data"]["class_labels"],
    model_name="XOR 6+6f KNN classifier",
    num_features=12,
    plot_index=np.array([0, 1]),
    fixed_features=np.ones(10) * 0.5,
    classifier_name="model"
)
proset.plot_decision_surface(
    features=result["data"]["X_test"][ix, :2],
    target=result["data"]["y_test"][ix],
    model=result["model"],
    feature_names=result["data"]["feature_names"][:2],
    class_labels=result["data"]["class_labels"],
    model_name="XOR 6+6f KNN classifier",
    use_proba=True,
    num_features=12,
    plot_index=np.array([0, 1]),
    fixed_features=np.ones(10) * 0.5,
    classifier_name="model"
)

print("* Done")
