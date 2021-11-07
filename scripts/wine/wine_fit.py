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
    ("wine_2d_05_model", {  # fix both alphas to 5 % and choose optimal penalty weights and number of batches
        "model_para": {"alpha_v": 0.05, "alpha_w": 0.05, "num_candidates": 1000},
        "select_para": {}
    }),
    ("wine_2d_50_model", {  # fix both alphas to 50 % and choose optimal penalty weights and number of batches
        "model_para": {"alpha_v": 0.50, "alpha_w": 0.50, "num_candidates": 1000},
        "select_para": {}
    }),
    ("wine_2d_95_model", {  # fix both alphas to 95 % and choose optimal penalty weights and number of batches
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {}
    }),
    ("wine_1d_model", {
        # fix both alphas to 95 % and the prototype penalty weight to the recommended default 1e-8; choose the feature
        # penalty weight and the number of batches
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {"lambda_w_range": 1e-8, "stage_1_trials": 11}  # need fewer trials for 1D search
    }),
    ("wine_fix_model", {
        # fix both alphas to 95 %, the feature penalty weight to 1e-3, and the prototype penalty weight to 1e-8; choose
        # the number of batches
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {"lambda_v_range": 1e-3, "lambda_w_range": 1e-8}
    }),
    ("wine_fix_opt_model", {
        # fix both alphas to 95 %; fix the feature penalty weight, prototype penalty weight, and number of batches to
        # the optimal values found for experiment wine_2d_95_model (not using 'one standard error rule')
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {
            "lambda_v_range": 3.81836792e-05, "lambda_w_range": 6.89780867e-09, "num_batch_grid": np.array([5])
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
    transform=StandardScaler(),
    random_state=random_state,
    **experiment[1]["select_para"]
)

print("* Save results")
result["data"] = data
with gzip.open(os.path.join(working_path, experiment[0] + ".gz"), mode="wb") as file:
    pickle.dump(result, file)

print("* Done")