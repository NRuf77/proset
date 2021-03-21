"""Profile proset classifier.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from cProfile import Profile
from io import StringIO
from pstats import Stats

import numpy as np
from sklearn.datasets import load_digits

from proset import ClassifierModel
from proset.benchmarks import start_console_log


start_console_log()


print("* Create data sample and model object")
random_state = np.random.RandomState(12345)
data = load_digits()
X = data["data"]
y = data["target"]
del data
model = ClassifierModel(random_state=random_state)

print("* Profile model fit")
profiler = Profile()
profiler.enable()
model.fit(X, y)
profiler.disable()
stream = StringIO()
profile_stats = Stats(profiler, stream=stream).strip_dirs().sort_stats("time")
profile_stats.print_stats(20)
print(stream.getvalue())
profile_stats.print_callees("quick_compute_similarity")
print(stream.getvalue())

print("* Done")
