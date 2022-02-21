# Python package proset
proset copyright 2022 by Nikolaus Ruf

Released under the MIT license - see LICENSE file for details

## About 
This package implements a supervised learning method we call the 'prototype set' or 'proset' algorithm.

The algorithm applies feature selection via an elastic net penalty [[1]](#1) to a nonlinear distribution model.
This uses locally weighted averaging similar to the extension of the Nadaraya-Watson estimator [[2]](#2)[[3]](#3) to conditional distributions [[4]](#4).
Instead of including a term for each training sample with unit weights, the algorithm selects a subset of representative samples (prototypes) with individual weights.
Prototype selection is handled via random subsampling and controlled by a second elastic net penalty.

Proset models are highly explainable due to their built-in feature selection and geometric properties:
- Feature selection makes it easier for humans to review the model structure.
If the number of relevant features is small, users can assess whether the choice is sensible and study low-dimensional representations like scatter plots or cuts through the decision space.
- Prototype selection simplifies reviewing the model structure even if the number of features is large.
We can perform weighted PCA on the feature matrix for the prototypes and use this to create low-dimensional maps of the data.
Also, a check whether the training data has labeling errors or artifacts can start with the smaller set of prototypes.
- The estimate for a particular sample can be explained by reviewing the prototypes with the highest impact.
This is an explanation in terms of similar training instances instead of more abstract properties, which can help nontechnical users to understand and trust the model.
- Proset rates new samples based on their absolute distance to the prototypes.
That means the algorithm can detect whether a new sample is far away from the training data and the estimate should not be relied on.

A technical report describing the algorithm in detail can be found here:

[> technical report (PDF)](https://github.com/NRuf77/proset/tree/master/doc/released/proset.pdf)

The report includes a benchmark study covering hyperparameter selection, comparison to other estimators, and explanatory features.

## Installation
Proset can be installed from PyPI via
```
pip install proset
```

This installs the package itself without the unit tests and benchmark scripts.
If you are interested in these, please clone or download the full source code from GitHub:

[> proset on github](https://github.com/NRuf77/proset)

### Dependencies
Proset requires Python 3.8 or later with the following packages:
- matplotlib >= 3.3.2
- numpy >= 1.19.2
- pandas >= 1.1.3
- scipy >= 1.5.2
- scikit-learn >= 0.23.2
- statsmodels >= 0.12.0

Additional packages are required to run the benchmark scripts:
- psutil >= 5.7.2
- shap >= 0.39.0
- xgboost >= 1.3.3

Use this command to install proset with additional dependencies:
```
pip install proset[benchmarks]
```

## Usage
Proset implements an interface compatible with machine learning package scikit-learn [[5]](#5).
You can create an estimator object like this:

```
from proset import ClassifierModel

model = ClassifierModel()
```

The model implements the fit(), predict(), predict_proba(), and score() methods required for scikit-learn estimators.
It has three additional public methods export(), explain(), and shrink().
The first creates a report with model parameters, the second explains a particular prediction, and the last reduces the model to expect only the active features as input.

To learn more about using proset, you can...
- use Python's help() to print the docstring for each function.
- refer to Chapter 5 'Implementation notes' of the technical report.
- look at the scripts for the benchmark study, which can serve as a tutorial:

[> benchmark scripts](https://github.com/NRuf77/proset/tree/master/scripts/)

## Release history
- version 0.1.0: implementation of proset classifier using algorithm L-BFGS-B [[6]](#6) for parameter estimation;
helper functions for model fitting and plotting;
benchmark code for hyperparameter selection, comparison to other classifiers, and demonstration of explanatory features;
first version of technical report.

### Note on performance
While we made some effort to implement computations efficiently, fitting a proset model takes more time than the other estimators in  the benchmark study.
This is not an issue for the smaller example cases with a few hundred samples, but cross-validation with a few thousand samples requires several hours of compute time on commodity hardware.
We intend to improve on this situation with further releases.

## Contact
Please contact <nikolaus.ruf@t-online.de> for any questions or suggestions.

## References
<a id="1">[1]</a> H. Zou, T. Hastie: 'Regularization and variable selection via the elastic net', Journal of the Royal Statistical Society, Series B, vol. 37, part 2, pp. 301-320, 2005.

<a id="2">[2]</a> E. A. Nadaraya: 'On Estimating Regression', Theory of Probability and Its Applications, vol. 9, issue 1, pp. 141-142, 1964.

<a id="3">[3]</a> G. S. Watson: 'Smooth Regression Analysis', Sankhyā: The Indian Journal of Statistics, Series A, vol. 26, no. 4, pp. 359-372, 1964.

<a id="4">[4]</a> P. Hall, J. Racine, Q. Li: 'Cross-validation and the Estimation of Conditional Probability Densities', Journal of the American Statistical Association, vol. 99, issue 468, pp. 1015-1026, 2004.

<a id="5">[5]</a> F. Pedregosa et al.: 'Scikit-learn: Machine Learning in Python', JMLR 12, pp. 2825-2830, 2011.

<a id="6">[6]</a> R. H. Byrd, P. Lu, J. Nocedal: 'A Limited Memory Algorithm for Bound Constrained Optimization', SIAM Journal on Scientific and Statistical Computing, vol. 16, issue 5, pp. 1190-1208, 1995.
