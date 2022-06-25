"""Prepare CIFAR-10 image data for transfer learning as benchmark case.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from proset.shared import FLOAT_TYPE, check_feature_names


# noinspection PyShadowingNames
def reshape_row(row):
    target = np.zeros((32, 32, 3), dtype=int)
    target[:, :, 0] = np.reshape(row[:1024], (32, 32))  # red
    target[:, :, 1] = np.reshape(row[1024:2048], (32, 32))  # green
    target[:, :, 2] = np.reshape(row[2048:], (32, 32))  # blue
    return target


print("* Apply user settings")
data_path = "scripts/data"
model_file = "resnet50.h5"
data_file = "cifar-10-python.tar.gz"  # downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
output_path = "scripts/results"
output_file = "cifar-10_data.gz"

print("* Initialize ResNet50")
model_path = os.path.join(data_path, model_file)
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
else:
    model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(32, 32, 3),
        pooling="avg"
        # as the input is of size 32-by-32, the first two dimensions of the output are 1-by-1; pooling only reduces the
        # dimensionality of the feature tensor
    )
    model.save(model_path)

print("* Process data")
train = []
test = None
label_names = None
with tarfile.open(name=os.path.join(data_path, data_file), mode="r:gz") as tar:
    for member in tar.getmembers():
        if "data_batch" in member.name:
            train.append(pickle.load(tar.extractfile(member), encoding="bytes"))
        elif "test" in member.name:
            test = pickle.load(tar.extractfile(member), encoding="bytes")
        elif "meta" in member.name:
            label_names = pickle.load(tar.extractfile(member), encoding="bytes")
X_train = np.vstack([batch[b"data"] for batch in train])
X_train = np.stack([reshape_row(X_train[i, :]) for i in range(X_train.shape[0])])
y_train = np.hstack([batch[b"labels"] for batch in train])
X_test = test[b"data"]
X_test = np.stack([reshape_row(X_test[i, :]) for i in range(X_test.shape[0])])
y_test = np.array(test[b"labels"], dtype=int)
label_names = [name.decode() for name in label_names[b"label_names"]]
data = {
    "X_train": model.predict(tf.keras.applications.resnet.preprocess_input(X_train)).astype(**FLOAT_TYPE),
    "X_test": model.predict(tf.keras.applications.resnet.preprocess_input(X_test)).astype(**FLOAT_TYPE),
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": np.array(check_feature_names(num_features=2048, feature_names=None, active_features=None)),
    "class_names": label_names
}

print("* Save data")
with gzip.open(os.path.join(output_path, output_file), mode="wb") as file:
    pickle.dump(data, file)

print("* Show sample to check conversion")
plt.figure()
plt.imshow(X=np.squeeze(X_train[0, :]))
plt.title("Image {}".format(label_names[y_train[0]]))

print("* Done")
