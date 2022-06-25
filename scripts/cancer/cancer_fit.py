"""Train proset classifier on breast cancer data.

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
data_file = "cancer_data.gz"
experiments = (
    ("cancer_2d_05_model", {  # fix both alphas to 5 % and choose optimal penalty weights and number of batches
        "model_para": {"alpha_v": 0.05, "alpha_w": 0.05, "num_candidates": 1000},
        "select_para": {}
    }),
    ("cancer_2d_50_model", {  # fix both alphas to 50 % and choose optimal penalty weights and number of batches
        "model_para": {"alpha_v": 0.50, "alpha_w": 0.50, "num_candidates": 1000},
        "select_para": {}
    }),
    ("cancer_2d_95_model", {  # fix both alphas to 95 % and choose optimal penalty weights and number of batches
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {}
    }),
    ("cancer_1d_model", {
        # fix both alphas to 95 % and the prototype penalty weight to the recommended default 1e-8; choose the feature
        # penalty weight and the number of batches
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {"lambda_w_range": 1e-8, "stage_1_trials": 11}  # need fewer trials for 1D search
    }),
    ("cancer_fix_model", {
        # fix both alphas to 95 %, the feature penalty weight to 1e-3, and the prototype penalty weight to 1e-8; choose
        # the number of batches
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {"lambda_v_range": 1e-3, "lambda_w_range": 1e-8}
    }),
    ("cancer_fix_opt_model", {
        # fix both alphas to 95 %; fix the feature penalty weight, prototype penalty weight, and number of batches to
        # the optimal values found for experiment cancer_2d_95_model (not using 'one standard error rule')
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {
            "lambda_v_range": 5.47739482e-03, "lambda_w_range": 8.30316381e-08, "num_batch_grid": np.array([1])
        }
    }),
    ("cancer_timing_1e7_model", {  # as cancer_2d_95_model with solver_factr fixed to 1e7 for timing purposes
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {"solver_factr": 1e7}
    }),
    ("cancer_timing_1e10_model", {  # as cancer_2d_95_model with solver_factr fixed to 1e7 for timing purposes
        "model_para": {"alpha_v": 0.95, "alpha_w": 0.95, "num_candidates": 1000},
        "select_para": {"solver_factr": 1e10}
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
