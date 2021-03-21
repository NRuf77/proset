"""Train proset classifier on breast cancer data.

Use optimal parameters instead of equivalent sparser solution, both alphas at 0.95 (dominant l2 penalty).

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
from psutil import cpu_count
import os
import pickle

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import proset
from proset.benchmarks import start_console_log


CPU_COUNT = cpu_count(logical=False)
NUM_JOBS = int(CPU_COUNT / 4) if CPU_COUNT >= 8 else None
# run cross-validation in parallel if at least 8 physical cores are available, assuming numpy uses 4 cores


if __name__ == "__main__":
    print("* Apply user settings")
    start_console_log()
    random_state = np.random.RandomState(12345)
    output_path = "scripts/results"
    output_file = "cancer_fix_opt_model.gz"

    print("* Load and format data")
    data = load_breast_cancer()
    X = data["data"]
    feature_names = data["feature_names"]
    y = data["target"]
    class_labels = data["target_names"]
    del data

    print("* Make train-test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    print("* Select hyperparameters via cross-validation")
    result = proset.select_hyperparameters(
        model=proset.ClassifierModel(alpha_v=0.95, alpha_w=0.95, num_candidates=1000),
        features=X_train,
        target=y_train,
        transform=StandardScaler(),
        lambda_v_range=6.66269950e-03,
        lambda_w_range=2.69938588e-07,
        num_batch_grid=np.array([2]),
        random_state=random_state,
        num_jobs=NUM_JOBS
    )

    print("* Save results")
    result["data"] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "class_labels": class_labels
    }
    with gzip.open(os.path.join(output_path, output_file), mode="wb") as file:
        pickle.dump(result, file)

    print("* Done")
