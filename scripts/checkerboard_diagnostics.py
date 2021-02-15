"""Score proset classifier trained on a deterministic checkerboard pattern.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

from sklearn.metrics import classification_report, log_loss, roc_auc_score

import proset


print("* Apply user settings")
input_path = "scripts/results"
input_file = "checkerboard_model.gz"

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
print("log-loss = {:.1e}".format(log_loss(y_true=truth, y_pred=probabilities)))
print("roc-auc  = {:.2f}\n".format(roc_auc_score(y_true=truth, y_score=probabilities[:, 1])))
print("- Classification report")
print(classification_report(y_true=truth, y_pred=prediction))

proset.plot_select_results(result)
proset.plot_decision_surface(
    features=result["data"]["X_test"],
    target=result["data"]["y_test"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="checkerboard classifier",
    classifier_name="model"
)
proset.plot_decision_surface(
    features=result["data"]["X_test"],
    target=result["data"]["y_test"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="checkerboard classifier",
    use_proba=True,
    classifier_name="model"
)

print("* Done")
