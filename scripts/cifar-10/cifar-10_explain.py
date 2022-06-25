"""Show prototypes with highest impact for one correctly and one incorrectly classified case.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle
import tarfile

import matplotlib.pyplot as plt
import numpy as np


# noinspection PyShadowingNames
def reshape_row(row):
    target = np.zeros((32, 32, 3), dtype=int)
    target[:, :, 0] = np.reshape(row[:1024], (32, 32))  # red
    target[:, :, 1] = np.reshape(row[1024:2048], (32, 32))  # green
    target[:, :, 2] = np.reshape(row[2048:], (32, 32))  # blue
    return target


def suppress_tick_labels():
    """Suppress tick labels on the current matplotlib axes.

    :return: no return value; axes formatted
    """
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])


print("* Apply user settings")
sample_class = "cat"
data_path = "scripts/data"
data_file = "cifar-10-python.tar.gz"  # downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
input_path = "scripts/results"
input_files = [
    "cifar-10_tf_model.gz",
    "cifar-10_pca_no_scaling_tf_model.gz",
    "cifar-10_pca_tf_model.gz",
]
print("  Select input file:")
for i, file_name in enumerate(input_files):
    print("  {} - {}".format(i, file_name))
choice = int(input())
input_file = input_files[choice]
model_name = input_file.replace(".gz", "")

print("* Load image data and model fit results")
train = []
test = None
with tarfile.open(name=os.path.join(data_path, data_file), mode="r:gz") as tar:
    for member in tar.getmembers():
        if "data_batch" in member.name:
            train.append(pickle.load(tar.extractfile(member), encoding="bytes"))
        elif "test" in member.name:
            test = pickle.load(tar.extractfile(member), encoding="bytes")
X_train = np.vstack([batch[b"data"] for batch in train])
X_train = np.stack([reshape_row(X_train[i, :]) for i in range(X_train.shape[0])])
y_train = np.hstack([batch[b"labels"] for batch in train])
X_test = test[b"data"]
X_test = np.stack([reshape_row(X_test[i, :]) for i in range(X_test.shape[0])])
y_test = np.array(test[b"labels"], dtype=int)
with gzip.open(os.path.join(input_path, input_file), mode="rb") as file:
    result = pickle.load(file)

print("* Find sample images")
sample_label = np.nonzero(np.array(result["data"]["class_names"]) == sample_class)[0][0]
test_features = result["model"]["transform"].transform(result["data"]["X_test"])
test_labels = result["data"]["y_test"]
prediction, familiarity = result["model"]["model"].predict(X=test_features, compute_familiarity=True)
probabilities = result["model"]["model"].predict_proba(test_features)
ix = np.nonzero(test_labels == sample_label)[0]
best_ix = ix[np.argmax(probabilities[ix, sample_label])]
worst_ix = ix[np.argmin(probabilities[ix, sample_label])]

print("* Find most relevant prototypes")
best_explain = result["model"]["model"].explain(
    X=test_features[best_ix:(best_ix + 1)], y=sample_label, familiarity=familiarity, include_features=False
)
best_familiarity = float(best_explain["sample name"].iloc[0][-4:])
best_prototypes = best_explain[["batch", "sample", "target"]].iloc[11:16]
best_prototypes["batch"] = best_prototypes["batch"].astype(int)
best_prototypes["sample"] = best_prototypes["sample"].astype(int)
best_prototypes["train_ix"] = [
    result["chunk_ix"][best_prototypes["batch"].iloc[i] - 1][best_prototypes["sample"].iloc[i]] for i in range(5)
]  # remap chunk index to original training data index
best_prototypes["contribution"] = [
    best_explain["p class {}".format(best_prototypes["target"].iloc[i])].iloc[11 + i] for i in range(5)
]
worst_explain = result["model"]["model"].explain(
    X=test_features[worst_ix:(worst_ix + 1)], y=sample_label, familiarity=familiarity, include_features=False
)
worst_familiarity = float(worst_explain["sample name"].iloc[0][-4:])
worst_prototypes = worst_explain[["batch", "sample", "target"]].iloc[11:16]
worst_prototypes["batch"] = worst_prototypes["batch"].astype(int)
worst_prototypes["sample"] = worst_prototypes["sample"].astype(int)
worst_prototypes["train_ix"] = [
    result["chunk_ix"][worst_prototypes["batch"].iloc[i] - 1][worst_prototypes["sample"].iloc[i]] for i in range(5)
]
worst_prototypes["contribution"] = [
    worst_explain["p class {}".format(worst_prototypes["target"].iloc[i])].iloc[11 + i] for i in range(5)
]
worst_probabilities = worst_explain[["p class {}".format(i) for i in range(10)]].iloc[0].values
worst_prediction = np.argmax(worst_probabilities)

print("* Show results")
plt.figure()
plt.subplot(2, 6, 1)
plt.imshow(X=np.squeeze(X_test[best_ix, :]))
suppress_tick_labels()
plt.title("Best classified {}\nfamiliarity = {:.2f}\nprobability = {:.2f}".format(
    sample_class, best_familiarity, probabilities[best_ix, sample_label]
))
plt.subplot(2, 6, 7)
plt.imshow(X=np.squeeze(X_test[worst_ix, :]))
suppress_tick_labels()
plt.title("Worst classified {}\nfamiliarity = {:.2f}\nprobability = {:.2f}\nprediction = {} ({:.2f})".format(
    sample_class,
    worst_familiarity,
    probabilities[worst_ix, sample_label],
    result["data"]["class_names"][worst_prediction],
    worst_probabilities[worst_prediction]
))
for i in range(5):
    plt.subplot(2, 6, 2 + i)
    plt.imshow(X=np.squeeze(X_train[best_prototypes["train_ix"].iloc[i], :]))
    suppress_tick_labels()
    plt.title("High impact prototype #{}\nclass = {}\ncontribution = {:.2f}".format(
        i + 1,
        result["data"]["class_names"][best_prototypes["target"].iloc[i]],
        best_prototypes["contribution"].iloc[i]
    ))
    plt.subplot(2, 6, 8 + i)
    plt.imshow(X=np.squeeze(X_train[worst_prototypes["train_ix"].iloc[i], :]))
    suppress_tick_labels()
    plt.title("High impact prototype #{}\nclass = {}\ncontribution = {:.2f}\n".format(
        i + 1,
        result["data"]["class_names"][worst_prototypes["target"].iloc[i]],
        worst_prototypes["contribution"].iloc[i]
    ))
plt.suptitle("CIFAR-10 classification example")
