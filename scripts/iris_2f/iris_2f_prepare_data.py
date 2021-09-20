"""Prepare two features of Fisher's iris data as benchmark case.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


print("* Apply user settings")
random_state = np.random.RandomState(12345)
output_path = "scripts/results"
output_file = "iris_2f_data.gz"

print("* Load and format data")
data = load_iris()
X = data["data"][:, :2]
feature_names = data["feature_names"][:2]
y = data["target_names"][data["target"]]  # use string target

print("* Make train-test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

print("* Save data")
data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": feature_names
}
with gzip.open(os.path.join(output_path, output_file), mode="wb") as file:
    pickle.dump(data, file)

print("* Done")
