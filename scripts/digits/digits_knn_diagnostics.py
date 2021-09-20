"""Score k-nearest neighbor classifier trained on the UCI ML hand-written digits dataset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

from sklearn.metrics import classification_report, log_loss, roc_auc_score


print("* Apply user settings")
input_path = "scripts/results"
input_file = "digits_knn_model.gz"

print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
test_labels = result["data"]["y_test"]
prediction = result["model"].predict(result["data"]["X_test"])
probabilities = result["model"].predict_proba(result["data"]["X_test"])
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

print("* Done")
