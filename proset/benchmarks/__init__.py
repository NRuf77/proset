"""Turn benchmarks into a sub-module of proset.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from proset.benchmarks.auxiliary import start_console_log
from proset.benchmarks.reference import fit_knn_classifier, fit_xgb_classifier, print_xgb_classifier_report
from proset.benchmarks.samples import create_checkerboard, create_continuous_xor
