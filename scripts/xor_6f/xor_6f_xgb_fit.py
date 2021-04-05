"""Train XGBoost classifier on the 'continuous XOR' problem with 6 features.

Optimize w.r.t. k.

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
data_path = "scripts/results"
input_file = "xor_6f_2d_95_model.gz"
output_file = "xor_6f_xgb_model.gz"

print("* Load and format data")
with gzip.open(os.path.join(data_path, input_file), mode="rb") as file:
    data = pickle.load(file)["data"]
# reuse train-test split from proset model fit

print("* Select hyperparameters via cross-validation")
result = fit_xgb_classifier(
    features=data["X_train"],
    labels=data["y_train"],
    max_depth=20,  # classifier is unable to engage with data for max_depth = 10
    colsample_range=(0.1, 0.9),
    subsample_range=(0.1, 0.9),
    num_folds=5,
    random_state=random_state
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(data_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
