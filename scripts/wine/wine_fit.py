"""Train proset classifier on wine classification data.

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
data_file = "wine_data.gz"
experiments = (
    ("wine_10b_model", {  # default parameters, 10 batches maximum
        "model_para": {},
        "select_para": {}
    }),
    ("wine_10b_beta_50_model", {  # change beta to 0.5
        "model_para": {"beta": 0.5},
        "select_para": {}
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
    transform=StandardScaler(),
    random_state=random_state,
    **experiment[1]["select_para"]
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, experiment[0] + ".gz"), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")
