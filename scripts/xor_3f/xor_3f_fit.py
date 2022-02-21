"""Train proset classifier on the 'continuous XOR' problem with 3 features.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import gzip
from psutil import cpu_count
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from proset import ClassifierModel
from proset.benchmarks import start_console_log
from proset.utility import select_hyperparameters


CPU_COUNT = cpu_count(logical=False)
NUM_JOBS = int(CPU_COUNT / 4) if CPU_COUNT >= 8 else None
# run cross-validation in parallel if at least 8 physical cores are available, assuming numpy uses 4 cores


if __name__ == "__main__":  # need import guard in case of parallel processing
    print("* Apply user settings")
    start_console_log()
    random_state = np.random.RandomState(12345)
    working_path = "scripts/results"
    data_file = "xor_3f_data.gz"
    output_file = "xor_3f_2d_95_model.gz"

    print("* Load data")
    with gzip.open(os.path.join(working_path, data_file), mode="rb") as file:
        data = pickle.load(file)

    print("* Select hyperparameters via cross-validation")
    result = select_hyperparameters(
        model=ClassifierModel(alpha_v=0.95, alpha_w=0.95, num_candidates=1000),
        features=data["X_train"],
        target=data["y_train"],
        transform=StandardScaler(),
        random_state=random_state,
        num_jobs=NUM_JOBS
    )

    print("* Save results")
    result["data"] = data
    with gzip.open(os.path.join(working_path, output_file), mode="wb") as file:
        pickle.dump(result, file)

    print("* Done")
