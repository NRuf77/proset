"""Train proset classifier on the 'continuous XOR' problem with 6 features.

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
data_file = "xor_6f_data.gz"
output_file = "xor_6f_50b_model.gz"

print("* Load data")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)

print("* Select hyperparameters via cross-validation")
result = select_hyperparameters(
    # increase max_batches to 50 as optimum was at 10 (20, 30) for 10 (20, 30) max_batches
    model=ClassifierModel(use_tensorflow=True),
    features=data["X_train"],
    target=data["y_train"],
    transform=StandardScaler(),
    max_batches=50,
    random_state=random_state
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
