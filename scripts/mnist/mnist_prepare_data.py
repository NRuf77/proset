"""Prepare MNIST digits data as benchmark case.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import matplotlib.pyplot as plt
import mnist
import numpy as np

from proset.shared import FLOAT_TYPE, check_feature_names


print("* Apply user settings")
mnist.temporary_dir = lambda: "scripts/data"
output_path = "scripts/results"
output_file = "mnist_data.gz"

print("* Process data")
data = {
    "X_train": np.reshape(mnist.train_images(), (60000, 28 ** 2)).astype(**FLOAT_TYPE),
    "X_test": np.reshape(mnist.test_images(), (10000, 28 ** 2)).astype(**FLOAT_TYPE),
    "y_train": mnist.train_labels(),
    "y_test": mnist.test_labels(),
    "feature_names": np.array(check_feature_names(num_features=28 ** 2, feature_names=None, active_features=None))
}

print("* Save data")
with gzip.open(os.path.join(output_path, output_file), mode="wb") as file:
    pickle.dump(data, file)

print("* Show sample to check conversion")
sample = 255 - np.reshape(data["X_train"][0, :], (28, 28))
plt.figure()
plt.imshow(X=sample, cmap="gray", vmin=0, vmax=255)
plt.title("Digit {}".format(data["y_train"][0]))

print("* Done")
