"""Train KNN classifier on the 'continuous XOR' problem with 3 features.

Optimize w.r.t. k.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from proset.benchmarks import fit_knn_classifier


print("* Apply user settings")
random_state = np.random.RandomState(12345)
data_path = "scripts/results"
input_file = "xor_3f_2d_95_model.gz"
output_file = "xor_3f_knn_model.gz"

print("* Load and format data")
with gzip.open(os.path.join(data_path, input_file), mode="rb") as file:
    data = pickle.load(file)["data"]
# reuse train-test split from proset model fit

print("* Select hyperparameters via cross-validation")
result = fit_knn_classifier(
    features=data["X_train"],
    labels=data["y_train"],
    transform=StandardScaler(),
    num_folds=5,
    random_state=random_state
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(data_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
