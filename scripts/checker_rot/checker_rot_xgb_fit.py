"""Train XGBoost classifier on the rotated checkerboard pattern.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np

from proset.benchmarks import fit_xgb_classifier


print("* Apply user settings")
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
data_file = "checker_rot_data.gz"
output_file = "checker_rot_xgb_model.gz"

print("* Load data")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)

print("* Select hyperparameters via cross-validation")
result = fit_xgb_classifier(features=data["X_train"], labels=data["y_train"], max_depth=30, random_state=random_state)
# trial with default depth 10 shows that larger trees may be needed

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
