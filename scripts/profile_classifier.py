"""Profile proset classifier.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from cProfile import Profile
import gzip
from io import StringIO
import logging
import os
import pickle
from pstats import Stats

import numpy as np
from sklearn.preprocessing import StandardScaler

from proset import ClassifierModel
from proset.benchmarks import start_console_log


print("* Apply user settings")
start_console_log(logging.DEBUG)
random_state = np.random.RandomState(12345)
working_path = "scripts/results"
experiments = (
    ("digits_np", {"data_file": "digits_data.gz", "use_tensorflow": False}),
    # fit model with one batch to digits data using numpy (only)
    ("digits_tf", {"data_file": "digits_data.gz", "use_tensorflow": True}),  # as above using tensorflow
    ("mnist_np", {"data_file": "mnist_data.gz", "use_tensorflow": False}),
    # fit model with one batch to mnist data using numpy (only)
    ("mnist_tf", {"data_file": "mnist_data.gz", "use_tensorflow": True})  # as above using tensorflow
)
print("  Select experiment")
for i in range(len(experiments)):
    print("  {} - {}".format(i, experiments[i][0]))
choice = int(input())
experiment = experiments[choice]

print("* Load data")
with gzip.open(os.path.join(working_path, experiment[1]["data_file"]), mode="rb") as file:
    data = pickle.load(file)

print("* Profile model fit")
model = ClassifierModel(random_state=random_state, use_tensorflow=experiment[1]["use_tensorflow"])
scaler = StandardScaler()
scaler.fit(data["X_train"])
profiler = Profile()
profiler.enable()
model.fit(X=scaler.transform(data["X_train"]), y=data["y_train"])
profiler.disable()
stream = StringIO()
profile_stats = Stats(profiler, stream=stream).strip_dirs().sort_stats("time")
profile_stats.print_stats(20)
print(stream.getvalue())
profile_stats.print_callees("_evaluate_objective")
print(stream.getvalue())

print("* Done")
