"""Train proset classifier on the MNIST digits dataset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

from proset import ClassifierModel
from proset.benchmarks import start_console_log
from proset.utility import select_hyperparameters


print("* Apply user settings")
start_console_log()
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
data_file = "mnist_data.gz"
output_file = "mnist_pca_10b_model.gz"

print("* Load data")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)

print("* Select hyperparameters via cross-validation")
result = select_hyperparameters(  # default parameters, 10 batches maximum, coarse grid due to time constraints
    model=ClassifierModel(num_candidates=10000, use_tensorflow=True),
    features=data["X_train"],
    target=data["y_train"],
    transform=PCA(n_components=0.99, whiten=True),  # PCA on unscaled input, scaled afterwards
    lambda_v_grid=np.logspace(-4.0, -2.0, 3),
    max_samples=40000,
    random_state=random_state
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
