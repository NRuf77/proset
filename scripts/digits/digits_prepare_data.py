"""Prepare UCI ML hand-written digits dataset as benchmark case.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from proset.shared import FLOAT_TYPE, check_feature_names


print("* Apply user settings")
random_state = np.random.RandomState(12345)
output_path = "scripts/results"
output_file = "digits_data.gz"

print("* Load and format data")
data = load_digits()
X = data["data"]
y = data["target"]
class_names = data["target_names"]
del data

print("* Make train-test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

print("* Save data")
data = {
    "X_train": X_train.astype(**FLOAT_TYPE),  # convert after split to retain F-contiguity
    "X_test": X_test.astype(**FLOAT_TYPE),
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": np.array(
        check_feature_names(num_features=X_train.shape[1], feature_names=None, active_features=None)
    )
}
with gzip.open(os.path.join(output_path, output_file), mode="wb") as file:
    pickle.dump(data, file)

print("* Done")
