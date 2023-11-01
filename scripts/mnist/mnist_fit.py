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
experiments = (
    ("mnist_pca_10b_model", {  # 10 batches maximum, use tensorflow
        "model_para": {"num_candidates": 10000, "use_tensorflow": True},
        "select_para": {"lambda_v_grid": np.logspace(-4.0, -2.0, 3), "max_samples": 40000}
    }),
    ("mnist_pca_10b_beta_50_model", {  # change beta to 0.5
        "model_para": {"num_candidates": 10000, "beta": 0.5, "use_tensorflow": True},
        "select_para": {"lambda_v_grid": np.logspace(-4.0, -2.0, 3), "max_samples": 40000}
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
    transform=PCA(n_components=0.99, whiten=True),  # PCA on unscaled input, scaled afterwards
    random_state=random_state,
    **experiment[1]["select_para"]
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, experiment[0] + ".gz"), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
