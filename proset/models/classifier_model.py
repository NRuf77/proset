"""Implementation of prototype set models with sklearn compatible interface.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted
from statsmodels.distributions.empirical_distribution import ECDF

from proset import shared
from proset.models.model import Model
from proset.objectives.np_classifier_objective import NpClassifierObjective
from proset.objectives.tf_classifier_objective import TfClassifierObjective
from proset.set_managers.classifier_set_manager import ClassifierSetManager


# noinspection PyPep8Naming, PyAttributeOutsideInit
class ClassifierModel(Model):
    """Prototype set classifier.
    """

    _estimator_type = "classifier"

    def _validate_y(self, y, reset):
        """Perform checks on classification target.

        :param y: list-like object; target for classification
        :param reset: see docstring of Model._validate_y() for details
        :return: 1D numpy integer array representing class labels as integers from 0 to K - 1; the model also gains two
            new properties:
            - classes_: 1D numpy array of original class labels
            - label_encoder_: object of type sklearn.preprocessing.LabelEncoder used to convert class labels to integers
        """
        check_classification_targets(y)
        if reset or not hasattr(self, "label_encoder_"):
            self.label_encoder_ = LabelEncoder()  # pylint: disable=attribute-defined-outside-init
            self.label_encoder_.fit(y)
            self.classes_ = self.label_encoder_.classes_  # pylint: disable=attribute-defined-outside-init
            # storing classes_ in the main estimator is an sklearn convention
        return self.label_encoder_.transform(y)

    @staticmethod
    def _get_compute_classes(use_tensorflow):
        """Provide classes implementing the set manager and objective function for the model.

        :param use_tensorflow: see docstring of Model._get_compute_classes()
        :return: subclasses of proset.set_manager.SetManager and proset.objective.Objective suitable for classification
        """
        if use_tensorflow:
            return ClassifierSetManager, TfClassifierObjective
        return ClassifierSetManager, NpClassifierObjective

    def _compute_prediction(self, X, n_iter, compute_familiarity):
        """Compute prediction.

        :param X: see docstring of Model._compute_prediction() for details
        :param n_iter: see docstring of Model._compute_prediction() for details
        :param compute_familiarity: see docstring of Model._compute_prediction() for details
        :return: as return value of Model._compute_prediction()
        """
        prediction = self.set_manager_.evaluate(
            features=X,
            num_batches=n_iter,
            prediction_type=shared.PredictionType.LIKELIHOOD,
            compute_familiarity=compute_familiarity
        )
        if compute_familiarity:
            familiarity = prediction[1]
            prediction = prediction[0]
        else:
            familiarity = None
        prediction = [self.classes_[np.argmax(p, axis=1)] for p in prediction]
        if isinstance(n_iter, np.ndarray):
            if compute_familiarity:
                return prediction, familiarity
            return prediction
        if compute_familiarity:
            return prediction[0], familiarity[0]
        return prediction[0]

    def _compute_score(self, X, y, sample_weight, n_iter):
        """Compute log-likelihood (not multiplied by -1 so it works with sklearn cross-validation).

        :param X: see docstring of Model._compute_score() for details
        :param y: 1D numpy integer array; class labels for classification as integers from 0 to K - 1
        :param sample_weight: see docstring of Model._compute_score() for details
        :param n_iter: see docstring of Model._compute_score() for details
        :return: as return value of Model._compute_score()
        """
        prediction = self.set_manager_.evaluate(
            features=X,
            num_batches=n_iter,
            prediction_type=shared.PredictionType.LIKELIHOOD,
            compute_familiarity=False
        )
        prediction = [np.squeeze(np.take_along_axis(p, y[:, None], axis=1)) for p in prediction]
        # keep only probability assigned to true class
        if sample_weight is None:
            prediction = [np.mean(np.log(p + shared.LOG_OFFSET)) for p in prediction]
        else:
            total_weight = np.sum(sample_weight)
            prediction = [np.sum(np.log(p + shared.LOG_OFFSET) * sample_weight) / total_weight for p in prediction]
        if isinstance(n_iter, np.ndarray):
            return np.array(prediction)
        return prediction[0]

    def predict_proba(self, X, n_iter=None, compute_familiarity=False):
        """Predict class probabilities for a feature matrix.

        :param X: 2D numpy array; feature matrix; sparse matrices or infinite/missing values not supported
        :param n_iter: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or None;
            number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for multiple
            values at once
        :param compute_familiarity: boolean; whether to compute the familiarity for each sample
        :return: 2D numpy array or list of 2D numpy arrays of type specified by shared.FLOAT_TYPE; each row contains the
            estimated class probabilities for the corresponding row in the feature matrix; if n_iter is integer or None,
            a single set of predictions is returned as an array; if n_iter is an array, a list of predictions is
            returned with one element for each element of the array; if compute_familiarity is True, also returns a 1D
            numpy float array of type specified by shared.FLOAT_TYPE or list of float arrays containing the familiarity
            of each sample
        """
        check_is_fitted(self, attributes="set_manager_")
        # noinspection PyUnresolvedReferences
        prediction = self.set_manager_.evaluate(
            features=check_array(X, **shared.FLOAT_TYPE),
            num_batches=n_iter,
            prediction_type=shared.PredictionType.LIKELIHOOD,
            compute_familiarity=compute_familiarity
        )
        if isinstance(n_iter, np.ndarray):
            if compute_familiarity:
                return prediction[0], prediction[1]
            return prediction
        if compute_familiarity:
            return prediction[0][0], prediction[1][0]
        return prediction[0]

    def _make_baseline_for_export(self):
        """Format marginal probabilities for export().

        :return: as return value of Model._make_baseline_for_export()
        """
        return pd.DataFrame({
            "batch": np.NaN,
            "sample": np.NaN,
            "sample name": self._format_class_labels(self.classes_),
            "target": range(self.classes_.shape[0]),
            # target column is numeric, look up class labels in sample name column for marginals
            "prototype weight": self.set_manager_.marginals
        }, columns=["batch", "sample", "sample name", "target", "prototype weight"])

    @staticmethod
    def _format_class_labels(classes):
        """Format class labels for report.

        :param classes: 1D numpy array; class labels
        :return: list of strings
        """
        return ["marginal probability class '{}'".format(label) for label in classes]

    def _make_contribution_report(self, prototype_report):
        """Format contribution of prototypes to prediction for report.

        :param prototype_report: see docstring of Model._make_contribution_report() for details
        :return: pandas data frame with the following fields:
            - dominant set: 0 or 1; indicates whether the prototype belongs to the dominant set
            - p class <class>: positive float; contribution of prototype to the estimated probability for the given
              class
        """
        contributions, dominant_set = self._compute_contributions(
            impact=prototype_report["impact"].to_numpy(copy=True),
            target=prototype_report["target"].to_numpy(copy=True),
            marginals=self.set_manager_.marginals
        )
        report = {"dominant set": dominant_set}
        columns = ["dominant set"]
        for i in range(self.classes_.shape[0]):
            column_name = "p class {}".format(i)
            report[column_name] = contributions[:, i]
            columns.append(column_name)
        return pd.DataFrame(report, columns=columns)

    @staticmethod
    def _compute_contributions(impact, target, marginals):
        """Compute contributions of prototypes to the probability estimate for the reference sample.

        :param impact: 1D numpy array of positive floats; impact of prototypes on the reference sample
        :param target: 1D numpy array of non-negative integers; target class corresponding to each prototype
        :param marginals: 1D numpy array of positive floats; marginal probabilities for each class
        :return: two return values:
            - 2D numpy array of non-negative floats; contribution of each prototype to the probability estimated with
              one row per prototype and one column per class
            - 1D numpy integer array; dominant set indicated by 1, other prototypes by 0
        """
        contributions = np.zeros((impact.shape[0], marginals.shape[0]), **shared.FLOAT_TYPE)
        scale = np.sum(impact) + 1.0
        if scale == 1.0:
            # the sample is so far away from the prototypes that the impact of prototypes is below numerical tolerance
            return contributions, np.zeros(contributions.shape[0], dtype=int)
        impact = impact / scale
        contributions[np.arange(contributions.shape[0]), target] = impact
        sort_ix = np.argsort(impact)[-1::-1]
        rank = rankdata(impact[sort_ix], method="dense")
        rank = np.hstack([0, 1 + np.max(rank) - rank])
        # the marginals have rank 0, all prototypes with the largest impact rank 1, etc.
        extended = np.vstack([marginals / scale, contributions[sort_ix, :]])
        extended = np.cumsum(np.add.reduceat(extended, indices=shared.find_changes(rank), axis=0), axis=0)
        # cumulative contribution to probability across ranks
        remainder = 1.0 - np.sum(extended, axis=1)  # remainder of probability missing up to a given rank
        top_two = np.array([np.sort(extended[i, :])[-1:-3:-1] for i in range(extended.shape[0])])
        # two larges probabilities assigned up to a given rank
        dominant_rank = np.nonzero(top_two[:, 0] - top_two[:, 1] <= remainder)[0]
        if len(dominant_rank) > 0:
            dominant_rank = dominant_rank[-1] + 1
            dominant_set = (rank[1:] <= dominant_rank).astype(int)
            # noinspection PyUnresolvedReferences
            dominant_set = dominant_set[np.argsort(sort_ix)]  # undo sorting
        else:
            dominant_set = np.zeros(contributions.shape[0], dtype=int)
        # noinspection PyUnresolvedReferences
        return contributions, dominant_set

    def _make_baseline_for_explain(
            self,
            X,
            y,
            n_iter,
            familiarity,
            sample_name,
            include_features,
            active_features,
            feature_columns,
            include_original,
            scale,
            offset
    ):
        """Format properties of baseline estimator for explain().

        :param X: see docstring of Model._make_baseline_for_explain() for details
        :param y: see docstring of Model._make_baseline_for_explain() for details
        :param n_iter: see docstring of Model._make_baseline_for_explain() for details
        :param familiarity: see docstring of Model._make_baseline_for_explain() for details
        :param sample_name: see docstring of Model._make_baseline_for_explain() for details
        :param include_features: see docstring of Model._make_baseline_for_explain() for details
        :param active_features: see docstring of Model._make_baseline_for_explain() for details
        :param feature_columns: see docstring of Model._make_baseline_for_explain() for details
        :param include_original: see docstring of Model._make_baseline_for_explain() for details
        :param scale: see docstring of Model._make_baseline_for_explain() for details
        :param offset: see docstring of Model._make_baseline_for_explain() for details
        :return: pandas data frame as return value of Model.explain(); the columns describing the contribution of
            prototypes to the classification result are
            - dominant set: integer; 1 indicates the prototype belongs to the dominant set, 0 that it does not
            - p class <class>: for the new sample, the predicted probability to belong to the given class; for other
              rows, the contribution towards that estimate
        """
        probabilities, sample_familiarity = self.predict_proba(X=X, n_iter=n_iter, compute_familiarity=True)
        prediction = self.label_encoder_.inverse_transform(np.array([np.argmax(probabilities)]))
        sample_name += ", prediction '{}'".format(prediction[0])
        if familiarity is not None:
            sample_name += ", familiarity {:.2f}".format(ECDF(familiarity)(sample_familiarity)[0])
        no_content = [np.NaN] * (self.classes_.shape[0] + 1)
        marginals = [np.NaN] + list(self.set_manager_.marginals)
        ones = [np.NaN] + [1.0] * (self.classes_.shape[0])
        report = {
            "batch": no_content,
            "sample": no_content,
            "sample name": [sample_name] + self._format_class_labels(self.classes_),
            "target": [
                self.label_encoder_.transform(np.array([y]))[0] if y is not None else np.NaN
            ] + list(range(self.classes_.shape[0])),
            # target column is numeric, look up class labels in sample name column for marginals
            "prototype weight": marginals,
            "similarity": ones,
            "impact": marginals,
            "dominant set": ones
        }
        columns = [
            "batch", "sample", "sample name", "target", "prototype weight", "similarity", "impact", "dominant set"
        ]
        for i in range(self.classes_.shape[0]):
            column_name = "p class {}".format(i)
            new_column = np.zeros(self.classes_.shape[0] + 1, **shared.FLOAT_TYPE)
            new_column[0] = probabilities[0, i]
            new_column[i + 1] = self.set_manager_.marginals[i] / (sample_familiarity[0] + 1.0)
            report[column_name] = new_column
            columns.append(column_name)
        if include_features:
            for i in active_features:
                report[feature_columns[i][0]] = no_content  # feature weight
                report[feature_columns[i][1]] = [X[0, i]] + no_content[:-1]  # feature value used by the model
                columns.extend(feature_columns[i][:2])
                if include_original:  # original feature value
                    report[feature_columns[i][2]] = [scale[i] * X[0, i] + offset[i]] + no_content[:-1]
                    columns.append(feature_columns[i][2])
                report[feature_columns[i][3]] = no_content  # per-feature similarity
                columns.append(feature_columns[i][3])
        return pd.DataFrame(report, columns=columns)
