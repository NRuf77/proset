"""Train proset classifier on the MNIST digits dataset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proset import ClassifierModel
from proset.benchmarks import start_console_log
from proset.utility import select_hyperparameters


print("* Apply user settings")
start_console_log()
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
data_file = "mnist_data.gz"
experiments = (
    ("mnist_tf_model", {
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 10000, "use_tensorflow": True},
        "select_para": {
            "transform": StandardScaler(),  # original features with scaling + centering only
            "lambda_w_range": 1e-8,
            "stage_1_trials": 11,
            "num_batch_grid": np.arange(21),
            "chunks": 2
        }
    }),
    ("mnist_pca_no_scaling_tf_model", {
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 10000, "use_tensorflow": True},
        "select_para": {
            "transform": PCA(n_components=0.99, whiten=True),  # PCA for unscaled features, scale output only
            "lambda_w_range": 1e-8,
            "stage_1_trials": 11,
            "num_batch_grid": np.arange(21),
            "chunks": 2
        }
    }),
    ("mnist_pca_tf_model", {
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 10000, "use_tensorflow": True},
        "select_para": {
            "transform": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.99, whiten=True))]),
            # PCA for scaled features, scale output again
            "lambda_w_range": 1e-8,
            "stage_1_trials": 11,
            "num_batch_grid": np.arange(21),
            "chunks": 2
        }
    })
)

print("  Select experiment")
for i in range(len(experiments)):
    print("  {} - {}".format(i, experiments[i][0]))
choice = int(input())
experiment = experiments[choice]

print("* Load data")
with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
    data = pickle.load(file)

print("* Select hyperparameters via cross-validation")
result = select_hyperparameters(
    model=ClassifierModel(**experiment[1]["model_para"]),
    features=data["X_train"],
    target=data["y_train"],
    random_state=random_state,
    **experiment[1]["select_para"]
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, experiment[0] + ".gz"), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
