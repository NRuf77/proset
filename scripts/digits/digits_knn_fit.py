"""Train KNN classifier on the UCI ML hand-written digits dataset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np

from proset.benchmarks import fit_knn_classifier


print("* Apply user settings")
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
data_file = "digits_data.gz"
output_file = "digits_knn_model.gz"

print("* Load data")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)

print("* Select hyperparameters via cross-validation")
result = fit_knn_classifier(
    features=data["X_train"], labels=data["y_train"], k_grid=np.arange(1, 101), random_state=random_state
)
# no scaling transform to avoid blowing up noise in areas that are almost uniformly white; trial with default shows more
# than 30 neighbors may be needed

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
