"""Train XGBoost classifier on two features of Fisher's iris data.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder

from proset.benchmarks import fit_xgb_classifier


print("* Apply user settings")
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
data_file = "iris_2f_data.gz"
output_file = "iris_2f_xgb_model.gz"

print("* Load")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)
encoder = LabelEncoder().fit(data["y_train"])

print("* Select hyperparameters via cross-validation")
t_start = time.time()
result = fit_xgb_classifier(
    features=data["X_train"],
    labels=encoder.transform(data["y_train"]),
    colsample_range=(0.1, 0.9),
    subsample_range=(0.1, 0.9),
    num_folds=5,
    random_state=random_state
)
t_end = time.time()
print("  - elapsed time = {} s".format(int(t_end - t_start)))

print("* Save results")
result["data"] = data
result["encoder"] = encoder
with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
