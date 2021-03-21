"""Score proset classifier trained on breast cancer data.

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
# input_file = "cancer_2d_05_model.gz"
# input_file = "cancer_2d_50_model.gz"
# input_file = "cancer_2d_95_model.gz"
# input_file = "cancer_1d_model.gz"
# input_file = "cancer_fix_model.gz"
input_file = "cancer_fix_opt_model.gz"

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

print("* Done")
