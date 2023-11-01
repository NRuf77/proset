"""Score proset classifier trained on the MNIST digits dataset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

from sklearn.metrics import classification_report, log_loss, roc_auc_score

import proset.utility as utility


print("* Apply user settings")
input_path = "scripts/results"
input_files = [
    "mnist_pca_10b_model.gz",
    "mnist_pca_10b_beta_50_model.gz"
]
print("  Select input file:")
for i, file_name in enumerate(input_files):
    print("  {} - {}".format(i, file_name))
choice = int(input())
input_file = input_files[choice]
model_name = input_file.replace(".gz", "")


print("* Load model fit results")
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Show results")
test_features = result["model"]["transform"].transform(result["data"]["X_test"])
test_labels = result["data"]["y_test"]
prediction = result["model"]["model"].predict(test_features)
probabilities = result["model"]["model"].predict_proba(test_features)
active_features = result["model"]["model"].set_manager_.get_active_features()
print("- Hyperparameter selection")
utility.print_hyperparameter_report(result)
print("-  Final model")
print("log-loss = {:.2f}".format(log_loss(y_true=test_labels, y_pred=probabilities)))
print("roc-auc  = {:.2f}".format(roc_auc_score(y_true=test_labels, y_score=probabilities, multi_class="ovo")))
print("number of input features = {}".format(result["model"]["model"].n_features_in_))
print("number of active features = {}".format(active_features.shape[0]))
print("degree of sparseness = {:.2f}".format(active_features.shape[0] / result["model"]["model"].n_features_in_))
print("number of prototypes = {}\n".format(result["model"]["model"].set_manager_.get_num_prototypes()))
print("- Classification report")
print(classification_report(y_true=test_labels, y_pred=prediction))
utility.plot_select_results(result=result, model_name=model_name)

print("* Done")
