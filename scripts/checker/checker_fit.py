"""Train proset classifier on a deterministic checkerboard pattern.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from proset import ClassifierModel
from proset.benchmarks import start_console_log
from proset.utility import select_hyperparameters


print("* Apply user settings")
start_console_log()
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
data_file = "checker_data.gz"
output_file = "checker_10b_model.gz"

print("* Load data")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)

print("* Select hyperparameters via cross-validation")
result = select_hyperparameters(  # default parameters, 10 batches maximum
    model=ClassifierModel(use_tensorflow=True),
    features=data["X_train"],
    target=data["y_train"],
    transform=StandardScaler(),
    random_state=random_state
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
