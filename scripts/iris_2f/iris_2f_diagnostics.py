"""Score proset classifier trained on two features of Fisher's iris data.

Uncomment the trial you want to see below.

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
# input_file = "iris_2f_2d_05_model.gz"
# input_file = "iris_2f_2d_50_model.gz"
# input_file = "iris_2f_2d_95_model.gz"
# input_file = "iris_2f_1d_model.gz"
# input_file = "iris_2f_fix_model.gz"
input_file = "iris_2f_fix_opt_model.gz"

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
print("roc-auc  = {:.2f}".format(roc_auc_score(y_true=truth, y_score=probabilities, multi_class="ovo")))
print("active features = {}".format(result["model"]["model"].set_manager_.num_active_features))
print("prototypes = {}\n".format(result["model"]["model"].set_manager_.num_prototypes))
print("- Classification report")
print(classification_report(y_true=truth, y_pred=prediction))

proset.plot_select_results(result)
proset.plot_decision_surface(
    features=result["data"]["X_test"],
    target=result["data"]["y_test"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="iris 2f classifier",
    classifier_name="model"
)
proset.plot_decision_surface(
    features=result["data"]["X_test"],
    target=result["data"]["y_test"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="iris 2f classifier",
    use_proba=True,
    classifier_name="model"
)
proset.plot_decision_surface(
    features=result["data"]["X_train"],
    target=result["data"]["y_train"],
    model=result["model"],
    feature_names=result["data"]["feature_names"],
    class_labels=result["data"]["class_labels"],
    model_name="iris 2f classifier (+ training samples)",
    use_proba=True,
    classifier_name="model"
)

print("* Done")
