"""Score proset classifier trained on the continuous XOR' problem with 3 features.

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
input_file = "xor_3f_2d_95_model.gz"

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
truth = result["data"]["y_test"]
prediction = result["model"].predict(result["data"]["X_test"])
probabilities = result["model"].predict_proba(result["data"]["X_test"])

print("- Hyperparameter selection")
proset.print_hyperparameter_report(result)
print("-  Final model")
print("log-loss = {:.2f}".format(log_loss(y_true=truth, y_pred=probabilities)))
print("roc-auc  = {:.2f}".format(roc_auc_score(y_true=truth, y_score=probabilities[:, 1])))
print("active features = {}".format(result["model"]["model"].set_manager_.num_active_features))
print("prototypes = {}\n".format(result["model"]["model"].set_manager_.num_prototypes))
print("- Classification report")
print(classification_report(y_true=truth, y_pred=prediction))

proset.plot_select_results(result)
ix = result["data"]["X_test"][:, 2] >= 0
# select test samples which have identical class based on the first two features
proset.plot_decision_surface(
    features=result["data"]["X_test"][ix, :2],
    target=result["data"]["y_test"][ix],
    model=result["model"],
    feature_names=result["data"]["feature_names"][:2],
    class_labels=result["data"]["class_labels"],
    model_name="XOR 3f classifier",
    num_features=3,
    plot_index=np.array([0, 1]),
    fixed_features=np.ones(1) * 0.5,
    classifier_name="model"
)
proset.plot_decision_surface(
    features=result["data"]["X_test"][ix, :2],
    target=result["data"]["y_test"][ix],
    model=result["model"],
    feature_names=result["data"]["feature_names"][:2],
    class_labels=result["data"]["class_labels"],
    model_name="XOR 3f classifier",
    use_proba=True,
    num_features=3,
    plot_index=np.array([0, 1]),
    fixed_features=np.ones(1) * 0.5,
    classifier_name="model"
)

print("* Done")
