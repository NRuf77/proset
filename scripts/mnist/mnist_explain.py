"""Some analyses regarding familiarity and 'easy' / 'hard' test cases.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sklearn.neighbors import KernelDensity


print("* Apply user settings")
input_path = "scripts/results"
input_files = [
    "mnist_tf_model.gz",
    "mnist_pca_no_scaling_tf_model.gz",
    "mnist_pca_tf_model.gz"
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
prediction, familiarity = result["model"]["model"].predict(X=test_features, compute_familiarity=True)
probabilities = result["model"]["model"].predict_proba(test_features)
print("-  Performance on 'hard' test cases")
print("log-loss = {:.2f}".format(log_loss(y_true=test_labels[:5000], y_pred=probabilities[:5000, :])))
print("roc-auc  = {:.2f}".format(
    roc_auc_score(y_true=test_labels[:5000], y_score=probabilities[:5000, :], multi_class="ovo"))
)
print("- Classification report")
print(classification_report(y_true=test_labels[:5000], y_pred=prediction[:5000]))
print("-  Performance on 'easy' test cases")
print("log-loss = {:.2f}".format(log_loss(y_true=test_labels[5000:], y_pred=probabilities[5000:, :])))
print("roc-auc  = {:.2f}".format(
    roc_auc_score(y_true=test_labels[5000:], y_score=probabilities[5000:, :], multi_class="ovo"))
)
print("- Classification report")
print(classification_report(y_true=test_labels[5000:], y_pred=prediction[5000:]))

print("* Compare familiarity for 'easy' and 'hard' cases")
data_range = (np.min(familiarity), np.max(familiarity))
delta = (data_range[1] - data_range[0]) / 20.0
grid = np.reshape(np.linspace(data_range[0] - delta, data_range[1] + delta, 1000), (1000, 1))
kde = KernelDensity(bandwidth=10.0)
kde.fit(familiarity[:5000, None])
kde_hard = np.exp(kde.score_samples(grid))
kde.fit(familiarity[5000:, None])
kde_easy = np.exp(kde.score_samples(grid))
plt.figure()
leg = [
    plt.plot(grid, kde_easy, 'b', linewidth=2)[0],
    plt.plot(grid, kde_hard, 'r', linewidth=2)[0]
]
plt.xlim([0.0, 400.0])  # adapted manually
plt.ylim([0.0, 0.01])
plt.grid("on")
plt.legend(leg, ["Easy cases", "Hard cases"])
plt.title("MNIST: familiarity distribution of 'easy' and 'hard' cases")
plt.xlabel("Familiarity")
plt.ylabel("Kernel density estimator")
plt.show()

print("* Compare misclassification rate for 'familiar' and 'unfamiliar' cases")
threshold = np.median(familiarity)
ix = familiarity < threshold
print("-  Performance on 'unfamiliar' test cases")
print("log-loss = {:.2f}".format(log_loss(y_true=test_labels[ix], y_pred=probabilities[ix, :])))
print("roc-auc  = {:.2f}".format(
    roc_auc_score(y_true=test_labels[ix], y_score=probabilities[ix, :], multi_class="ovo"))
)
print("- Classification report")
print(classification_report(y_true=test_labels[ix], y_pred=prediction[ix]))
ix = np.logical_not(ix)
print("-  Performance on 'familiar' test cases")
print("log-loss = {:.2f}".format(log_loss(y_true=test_labels[ix], y_pred=probabilities[ix, :])))
print("roc-auc  = {:.2f}".format(
    roc_auc_score(y_true=test_labels[ix], y_score=probabilities[ix, :], multi_class="ovo"))
)
print("- Classification report")
print(classification_report(y_true=test_labels[ix], y_pred=prediction[ix]))

print("* Done")
