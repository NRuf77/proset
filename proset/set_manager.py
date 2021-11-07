"""Implementation of set managers for prototype set models.

Copyright by Nikolaus Ruf
Released under the MIT license - see LICENSE file for details
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances

import proset.shared as shared


MERGE_TOL = 1e-8
# two prototypes are considered identical and suitable for merging if the maximum absolute difference across all feature
# and target values is at most equal to this value


class SetManager(metaclass=ABCMeta):
    """Abstract base class for set managers.
    """

    def __init__(self, target):
        """Initialize set manager.

        :param target: list-like object; target for supervised learning
        """
        self._batches = []
        self._meta = {
            # track properties of the fitting problem that may depend on subclass implementation; this is passed to all
            # static methods called by other public methods than __init__ in case overriding requires additional
            # information
            "num_features": None  # set when adding batches
        }
        # noinspection PyTypeChecker
        self._meta.update(self._get_baseline_distribution(target))

    @staticmethod
    @abstractmethod
    def _get_baseline_distribution(target):  # pragma: no cover
        """Compute baseline distribution parameters from target for supervised learning.

        :param target: see docstring of __init__() for details
        :return: dict; contains information regarding the baseline information that depends on the model
        """
        return NotImplementedError(
            "Abstract base class SetManager has no default implementation for method _get_baseline_distribution()."
        )

    @property
    def num_batches(self):
        """Get number of batches already added to SetManager instance.

        :return: integer; number of batches
        """
        return len(self._batches)

    @property
    def num_features(self):
        """Get number of features expected for input matrices.

        :return: integer or None; expected number of features; None if no batch has been added yet
        """
        return self._meta["num_features"]

    def get_active_features(self, num_batches=None):
        """Get indices of active feature across all batches.

        :param num_batches: non-negative integer or None; number of batches to use for evaluation; pass None for all
            batches
        :return: 1D numpy array of non-negative integers; active feature indices w.r.t. original feature matrix
        """
        num_batches = self._check_num_batches(
            num_batches=num_batches, num_batches_actual=self.num_batches, permit_array=False
        )
        active_features = [
            self._batches[i]["active_features"] for i in range(num_batches) if self._batches[i] is not None
        ]
        if len(active_features) == 0:
            return np.zeros(0, dtype=int)
        return np.unique(np.hstack(active_features))

    @staticmethod
    def _check_num_batches(num_batches, num_batches_actual, permit_array):
        """Check requested number of batches is consistent with actual number.

        :param num_batches: non-negative integer or 1D numpy array of strictly increasing, non-negative integers; number
            of batches for which computation is requested
        :param num_batches_actual: non-negative integer; actual number of batches
        :param permit_array: boolean; whether passing an array for num_batches is permissible
        :return: num_batches or num_batches_actual if the former is None; raises an error if a check fails
        """
        if isinstance(num_batches, np.ndarray):
            if not permit_array:
                raise TypeError("Parameter num_batches must not be an array.")
            if len(num_batches.shape) != 1:
                raise ValueError("Parameter num_batches must be 1D if passing an array.")
            if not np.issubdtype(num_batches.dtype, np.integer):
                raise TypeError("Parameter num_batches must be of integer type if passing an array.")
            if np.any(num_batches < 0):
                raise ValueError("Parameter num_batches must not contain negative values if passing an array.")
            if np.any(num_batches > num_batches_actual):
                raise ValueError(
                    " ".join([
                        "Parameter num_batches must not contain values greater than the available number of",
                        "batches ({}) if passing an array.".format(num_batches_actual)
                    ])
                )
            if np.any(np.diff(num_batches) <= 0):
                raise ValueError(
                    "Parameter num_batches must contain strictly increasing elements if passing an array."
                )
            num_batches = num_batches.copy()  # output should not be a reference to input
        else:
            if num_batches is None:
                num_batches = num_batches_actual
            if not np.issubdtype(type(num_batches), np.integer):
                raise TypeError("Parameter num_batches must be an integer.")
            if num_batches < 0:
                raise ValueError("Parameter num_batches must not be negative.")
            if num_batches > num_batches_actual:
                raise ValueError(
                    "Parameter num_batches must be less than or equal to the available number of batches ({}).".format(
                        num_batches_actual
                    ))
        return num_batches

    def get_num_prototypes(self):
        """Get number of prototypes across all batches.

        The same training sample is counted multiple times if it appears in multiple batches.

        :return: integer; number of prototypes
        """
        return np.sum([batch["scaled_prototypes"].shape[0] for batch in self._batches if batch is not None])

    def add_batch(self, batch_info):
        """Add batch of prototypes.

        :param batch_info: dict with the following fields:
            - prototypes: 2D numpy float array; feature matrix of prototypes; sparse matrices or infinite/missing values
              not supported
            - target: 1D numpy array; target for supervised learning; must have as many elements as prototypes has rows
            - feature_weights: 1D numpy float array; feature_weights for the batch; must have as many elements as
              prototypes has columns; elements must not be negative
            - prototype_weights: 1D numpy float array; prototype weights for the batch; must have as many elements as
              prototypes has rows; elements must not be negative
            - sample_index: 1D numpy integer array; indices of prototypes in training sample; must have as many elements
              as prototypes has rows
        :return: no return arguments; internal state updated with new batch
        """
        self._meta["num_features"] = self._check_batch(batch_info=batch_info, meta=self._meta)
        self._batches.append(self._process_batch(batch_info))

    @staticmethod
    def _check_batch(batch_info, meta):
        """Check batch definition for consistent dimensions.

        :param batch_info: see docstring of add_batch() for details
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet
        :return: integer; number of features; raises a ValueError if a check fails
        """
        if len(batch_info["prototypes"].shape) != 2:
            raise ValueError("Parameter prototypes must be a 2D array.")
        if meta["num_features"] is not None and batch_info["prototypes"].shape[1] != meta["num_features"]:
            raise ValueError("Parameter prototypes has {} columns but {} are expected.".format(
                batch_info["prototypes"].shape[1], meta["num_features"]
            ))
        if len(batch_info["target"].shape) != 1:
            raise ValueError("Parameter target must be a 1D array.")
        if batch_info["target"].shape[0] != batch_info["prototypes"].shape[0]:
            raise ValueError("Parameter target must have as many elements as prototypes has rows.")
        if len(batch_info["feature_weights"].shape) != 1:
            raise ValueError("Parameter feature_weights must be a 1D array.")
        if batch_info["feature_weights"].shape[0] != batch_info["prototypes"].shape[1]:
            raise ValueError("Parameter feature_weights must have as many elements as prototypes has columns.")
        if len(batch_info["prototype_weights"].shape) != 1:
            raise ValueError("Parameter prototype_weights must be a 1D array.")
        if batch_info["prototype_weights"].shape[0] != batch_info["prototypes"].shape[0]:
            raise ValueError("Parameter prototype_weights must have as many elements as prototypes has rows.")
        if len(batch_info["sample_index"].shape) != 1:
            raise ValueError("Parameter sample_index must be a 1D array.")
        if not np.issubdtype(batch_info["sample_index"].dtype, np.integer):
            raise TypeError("Parameter sample_index must be an integer array.")
        if batch_info["sample_index"].shape[0] != batch_info["prototypes"].shape[0]:
            raise ValueError("Parameter sample_index must have as many elements as prototypes has rows.")
        return batch_info["prototypes"].shape[1]

    # noinspection PyUnusedLocal
    @staticmethod
    def _process_batch(batch_info):
        """Process batch information.

        :param batch_info: see docstring of add_batch() for details
        :return: dict or None; returns None if all prototype weights are zero as the batch has no impact on the model;
            dict contains the information describing the batch in reduced form, taking advantage of sparseness; the
            following keys and values are included:
            - active_features: 1D numpy integer array; index vector of features with non-zero feature weights
            - scaled_prototypes: 2D numpy float array; prototypes reduced to active prototypes and features, scaled with
                feature weights
            - ssq_prototypes: 1D numpy float array; row sums of scaled prototypes
            - target: 1D numpy array; target values corresponding to scaled prototypes
            - feature_weights: 1D numpy float array; feature weights reduced to active features
            - prototype_weights: 1D numpy float array; prototype weights reduced to active prototypes
            - sample_index: 1D numpy integer array; sample indices reduced to active prototypes
        """
        if np.all(batch_info["prototype_weights"] == 0.0):
            return None
        active_features = np.nonzero(batch_info["feature_weights"] > 0.0)[0]
        feature_weights = batch_info["feature_weights"][active_features]
        active_prototypes = np.nonzero(batch_info["prototype_weights"] > 0.0)[0]
        scaled_prototypes = batch_info["prototypes"][active_prototypes][:, active_features] * feature_weights
        target = batch_info["target"][active_prototypes]
        prototype_weights = batch_info["prototype_weights"][active_prototypes]
        sample_index = batch_info["sample_index"][active_prototypes]
        relation = np.nonzero(
            pairwise_distances(X=np.hstack([scaled_prototypes, target[:, np.newaxis]]), metric="chebyshev") <= MERGE_TOL
        )  # find all pairs of features and target that are identical within tolerance
        num_labels, labels = connected_components(
            csgraph=sparse.coo_matrix((np.ones_like(relation[0]), (relation[0], relation[1]))),
            directed=False,
            return_labels=True
        )  # label groups of identical feature/target combinations
        if num_labels < len(labels):  # one or more prototypes can be merged together
            sort_ix = np.lexsort([sample_index, labels])
            # reduceat() requires equivalent prototypes to be grouped together; lexsort uses the last key as primary
            # sort key; using sample index as secondary key means the smallest index in each group is first
            changes = np.hstack([0, np.nonzero(np.diff(labels[sort_ix]))[0] + 1])
            scaled_prototypes = scaled_prototypes[sort_ix][changes]
            target = target[sort_ix][changes]
            prototype_weights = np.add.reduceat(prototype_weights[sort_ix], indices=changes, axis=0)
            sample_index = sample_index[sort_ix][changes]
        return {
            "active_features": active_features,
            "scaled_prototypes": scaled_prototypes,
            "ssq_prototypes": np.sum(scaled_prototypes ** 2.0, axis=1),
            "target": target,
            "feature_weights": feature_weights,
            "prototype_weights": prototype_weights,
            "sample_index": sample_index
        }

    def evaluate_unscaled(self, features, num_batches):
        """Compute unscaled predictions and scaling vector.

        :param features: 2D numpy float array; feature matrix for which to compute unscaled predictions and scales
        :param num_batches: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or
            None; number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for
            multiple values of num_batches at once
        :return: list of tuples; each tuple consists of two numpy arrays; the first array is either 1D or 2D and
            has the same first dimension as features; it represents the unscaled predictions (class probabilities or
            regression means); the second array is the corresponding scaling vector; if an integer is passed for
            num_batches, the list has length 1; else, the list has one element per element of num_batches
        """
        num_batches = self._check_evaluate_input(
            features=features,
            num_batches=num_batches,
            num_batches_actual=self.num_batches,
            permit_array=True,
            meta=self._meta
        )
        impact, target, batch_index = self._compute_impact(
            features=features,
            batches=self._batches,
            num_batches=num_batches[-1] if isinstance(num_batches, np.ndarray) else num_batches,
            # evaluate impact up the batch with the largest index requested
            meta=self._meta
        )
        return self._convert_to_unscaled(
            impact=impact,
            target=target,
            batch_index=batch_index,
            num_batches=num_batches,
            meta=self._meta
        )

    @classmethod
    def _check_evaluate_input(cls, features, num_batches, num_batches_actual, permit_array, meta):
        """Check whether input to evaluate_unscaled() is consistent.

        :param features: see docstring of evaluate_unscaled() for details
        :param num_batches: see docstring of evaluate_unscaled() for details
        :param num_batches_actual: non-negative integer; actual number of batches
        :param permit_array: boolean; whether passing an array for num_batches is permissible
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet
        :return: num_batches or num_batches_actual if the former is None; raises an error if a check fails
        """
        if len(features.shape) != 2:
            raise ValueError("Parameter features must be a 2D array.")
        if meta["num_features"] is not None and features.shape[1] != meta["num_features"]:
            # evaluate_unscaled() can be called before any batches have been fitted to get the default model
            raise ValueError("Parameter features has {} columns but {} are expected.".format(
                features.shape[1], meta["num_features"]
            ))
        return cls._check_num_batches(
            num_batches=num_batches, num_batches_actual=num_batches_actual, permit_array=permit_array
        )

    @staticmethod
    def _compute_impact(features, batches, num_batches, meta):
        """Compute impact of each prototype on each sample.

        :param features: see docstring of evaluate_unscaled() for details
        :param batches: list of dicts as generated by _process_batch()
        :param num_batches: non-negative integer; number of batches used for computation
        :param meta: dict; must contain key 'num_features' referencing number of features, unless batches is the empty
            list or num_batches is 0
        :return: three numpy arrays:
            - 2D array of positive floats with one row per sample and one column per prototype from the batches used;
              contains the impact of each prototype on the prediction of each sample
            - 1D array with target values for the prototypes
            - 1D array of non-negative integers; batch index
        """
        if len(batches) == 0 or num_batches == 0:  # default model is independent of prototypes
            return np.zeros((features.shape[0], 0), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
            # dtype=float is wrong for a classifier target but that does not matter for empty matrices
        impact = []
        target = []
        batch_index = []
        for i in range(num_batches):
            if batches[i] is not None:
                if batches[i]["active_features"].shape[0] == 0:  # no active features means a global adjustment
                    new_impact = np.tile(batches[i]["prototype_weights"], (features.shape[0], 1))
                else:
                    if batches[i]["active_features"].shape[0] == meta["num_features"]:  # no need to reduce input
                        scaled_features = features * batches[i]["feature_weights"]
                    else:  # reduce input to active features
                        scaled_features = features[:, batches[i]["active_features"]] * batches[i]["feature_weights"]
                        # broadcast scaling across rows
                    new_impact = shared.quick_compute_similarity(
                        scaled_reference=scaled_features,
                        scaled_prototypes=batches[i]["scaled_prototypes"],
                        ssq_reference=np.sum(scaled_features ** 2.0, axis=1),
                        ssq_prototypes=batches[i]["ssq_prototypes"]
                    ) * batches[i]["prototype_weights"]
                impact.append(new_impact)
                target.append(batches[i]["target"])
                batch_index.append(i * np.ones(new_impact.shape[1], dtype=int))
        if len(impact) == 0:  # all batches are empty
            return np.zeros((features.shape[0], 0), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
        return np.hstack(impact), np.hstack(target), np.hstack(batch_index)

    @classmethod
    @abstractmethod
    def _convert_to_unscaled(cls, impact, target, batch_index, num_batches, meta):  # pragma: no cover
        """Convert impact and target to unscaled predictions and scaling vector.

        :param impact: as first return value of _compute_impact()
        :param target: as second return value of _compute_impact()
        :param batch_index: as third return value of _compute_impact()
        :param num_batches: non-negative integer or 1D numpy array of strictly increasing integers; number of batches to
            use for evaluation; pass an array to evaluate for multiple values of num_batches at once
        :param meta: dict; properties of the fitting problem that may depend on the subclass
        :return: as return argument of evaluate_unscaled()
        """
        raise NotImplementedError(
            "Abstract base class SetManager has no default implementation for method _compute_unscaled()."
        )

    def evaluate(self, features, num_batches, compute_familiarity):
        """Compute scaled predictions.

        :param features: 2D numpy float array; feature matrix for which to compute scaled predictions
        :param num_batches: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or
            None; number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for
            multiple values of num_batches at once
        :param compute_familiarity: boolean; whether to compute the familiarity for each sample
        :return: list of numpy arrays; each array is either 1D or 2D with the same first dimension as features and
            contains predictions (class probabilities or regression means); if an integer is passed for num_batches, the
            list has length 1; else, the list has one element per element of num_batches
        """
        unscaled = self.evaluate_unscaled(features, num_batches)
        scaled = [(pair[0].transpose() / pair[1]).transpose() for pair in unscaled]
        # transpose to broadcast scale over columns in case unscaled is 2D
        if compute_familiarity:
            return scaled, [pair[1] - 1.0 for pair in unscaled]
        return scaled

    def get_feature_weights(self, num_batches=None):
        """Get weights of active features for all batches as a matrix.

        :param num_batches: non-negative integer or None; number of batches to export; pass None for all batches
        :return: dict with keys:
            - weight_matrix: 2D numpy float array; this has one row per batch and once column per feature that is active
              in at least one batch; features are sorted in order of descending weight for the first row, using
              subsequent rows as tie-breaker
            - feature_index: 1D numpy integer array; index vector indicating the order of features
        """
        num_batches = self._check_num_batches(
            num_batches=num_batches, num_batches_actual=self.num_batches, permit_array=False
        )
        active_features = self.get_active_features(num_batches)
        if active_features.shape[0] == 0:
            return {
                "weight_matrix": np.zeros((num_batches, 0)),
                "feature_index": np.zeros(0, dtype=int)
            }
        weight_matrix = []
        for i in range(num_batches):
            new_row = np.zeros(len(active_features))
            if self._batches[i] is not None:
                new_row[
                    np.searchsorted(active_features, self._batches[i]["active_features"])
                ] = self._batches[i]["feature_weights"]
            weight_matrix.append(new_row)
        order = np.lexsort(weight_matrix[-1::-1])[-1::-1]
        # np.lexsort() uses the last argument as primary key and sorts in ascending order
        return {
            "weight_matrix": np.row_stack(weight_matrix)[:, order],
            "feature_index": active_features[order]
        }

    def get_batches(self, features=None, num_batches=None):
        """Get batch information.

        :param features: 2D numpy float array with a single row or None; if not None, per-feature similarities are
            computed between features and each prototype
        :param num_batches: non-negative integer or None; number of batches to export; pass None for all batches
        :return: list whose elements are either dicts or None; None indicates the batch in this position contains no
            prototypes; each dict has the following fields:
            - active_features: 1D numpy integer array; index vector of features with non-zero feature weights
            - prototypes: 2D numpy float array; prototypes reduced to active features
            - target: 1D numpy array; target values corresponding to scaled prototypes
            - feature_weights: 1D numpy float array; feature weights reduced to active features
            - prototype_weights: 1D numpy float array; prototype weights
            - sample_index: 1D numpy integer array; sample indices for prototypes
            - similarities: 2D numpy array; per-feature similarities between the input features and each prototype; one
              row per prototype and one column per active feature; this field is not included if features is None
        """
        num_batches, features = self._check_get_batches_input(
            features=features,
            num_batches=num_batches,
            num_batches_actual=self.num_batches,
            meta=self._meta
        )
        batches = [{
            "active_features": self._batches[i]["active_features"].copy(),
            "prototypes":
                self._batches[i]["scaled_prototypes"].copy() if self._batches[i]["feature_weights"].shape[0] == 0 else
                self._batches[i]["scaled_prototypes"] / self._batches[i]["feature_weights"],
            "target": self._batches[i]["target"].copy(),
            "feature_weights": self._batches[i]["feature_weights"].copy(),
            "prototype_weights": self._batches[i]["prototype_weights"].copy(),
            "sample_index": self._batches[i]["sample_index"].copy()
        } if self._batches[i] is not None else None for i in range(num_batches)]
        if features is not None:
            for batch in batches:
                if batch is not None:
                    batch["similarities"] = self._compute_feature_similarities(
                        prototypes=batch["prototypes"],
                        features=features[batch["active_features"]],
                        feature_weights=batch["feature_weights"]
                    )
        return batches

    @classmethod
    def _check_get_batches_input(cls, features, num_batches, num_batches_actual, meta):
        """Check whether input to get_batches() is consistent.

        :param features: see docstring of get_batches() for details
        :param num_batches: see docstring of get_batches() for details
        :param num_batches_actual: non-negative integer; actual number of batches
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet
        :return: two return values:
            - non-negative integer; num_batches or num_batches_actual if the former is None
            - 1D numpy float array or None; features converted to a 1D array if not None
            raises an error if a check fails
        """
        if features is None:
            num_batches = cls._check_num_batches(
                num_batches=num_batches,
                num_batches_actual=num_batches_actual,
                permit_array=False
            )
        else:
            num_batches = cls._check_evaluate_input(
                features=features,
                num_batches=num_batches,
                num_batches_actual=num_batches_actual,
                permit_array=False,
                meta=meta
            )
            if features.shape[0] != 1:
                raise ValueError("Parameter features must have exactly one row.")
            features = np.squeeze(features)
        return num_batches, features

    @staticmethod
    def _compute_feature_similarities(prototypes, features, feature_weights):
        """Compute per-feature similarities between prototypes and a single reference sample.

        :param prototypes: 2D numpy float array; prototypes
        :param features: 1D numpy float array; features for reference sample
        :param feature_weights: 1D numpy array of non-negative floats; feature weights
        :return: as the value for key 'similarities' in the output of get_batches()
        """
        return np.exp(-0.5 * ((prototypes - features) * feature_weights) ** 2.0)

    def shrink(self):
        """Reduce internal state representation to active features across all batches.

        :return: 1D numpy array of non-negative integers; indices of active features w.r.t. original training data
        """
        active_features = self.get_active_features()
        if self._meta["num_features"] is None:  # nothing to do as no batches were ever added
            return active_features  # this is a vector of length zero by default
        self._meta["num_features"] = active_features.shape[0]
        for i in range(len(self._batches)):
            if self._batches[i] is not None:
                self._batches[i]["active_features"] = np.searchsorted(
                    active_features, self._batches[i]["active_features"]
                )  # locate batch active features among all active features
        return active_features


class ClassifierSetManager(SetManager):
    """Set manager class for proset classifier
    """

    @staticmethod
    def _get_baseline_distribution(target):
        """Compute baseline distribution parameters from target for classification.

        :param target: see docstring of __init__() for details
        :return: dict; contains information regarding the baseline information that depends on the model
        """
        counts = shared.check_classifier_target(target)
        return {"marginals": counts / np.sum(counts)}

    @property
    def marginals(self):
        """Get marginal probabilities.

        :return: 1D numpy float array; marginal class probabilities as values in (0.0, 1.0)
        """
        # noinspection PyUnresolvedReferences
        return self._meta["marginals"].copy()

    @staticmethod
    def _check_batch(batch_info, meta):
        """Check batch definition for consistent dimensions.

        :param batch_info: see docstring of SetManager.add_batch() for details
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet; must have key
            'marginals' containing a 1D numpy float array with the marginal distribution of the classes
        :return: integer; number of features; raises a ValueError if a check fails
        """
        if not np.issubdtype(batch_info["target"].dtype, np.integer):
            raise TypeError("Parameter target must have integer elements.")
        if np.any(batch_info["target"] < 0) or np.any(batch_info["target"] >= meta["marginals"].shape[0]):
            raise ValueError("Parameter target must encode the classes as integers from 0 to K - 1.")
        return SetManager._check_batch(batch_info=batch_info, meta=meta)

    @classmethod
    def _convert_to_unscaled(cls, impact, target, batch_index, num_batches, meta):
        """Convert impact and target to unscaled class probabilities and scaling vector.

        :param impact: as first return value of SetManager._compute_impact()
        :param target: as second return value of SetManager._compute_impact()
        :param batch_index: as third return value of SetManager._compute_impact()
        :param num_batches: non-negative integer or 1D numpy array of strictly increasing integers; number of batches to
            use for evaluation; pass an array to evaluate for multiple values of num_batches at once
        :param meta: dict; must have key 'marginals' referencing a 1D numpy float array of marginal class probabilities
        :return: as return argument of SetManager.evaluate_unscaled()
        """
        if not isinstance(num_batches, np.ndarray):
            num_batches = np.array([num_batches])
        sort_ix = np.argsort(target)
        # reduceat() used by _compute_update() requires prototypes with the same target value grouped together
        impact = impact[:, sort_ix]
        target = target[sort_ix]
        probabilities = np.tile(meta["marginals"], (impact.shape[0], 1))
        if num_batches.shape[0] == 1:
            if impact.shape[1] > 0:
                update, cols = cls._compute_update(impact, target)
                probabilities[:, cols] += update
            return [(probabilities, np.sum(probabilities, axis=1))]
        batch_index = batch_index[sort_ix]
        collect = []
        for i in range(num_batches.shape[0]):
            batch_from = num_batches[i - 1] if i > 0 else 0
            use_ix = np.logical_and(batch_index >= batch_from, batch_index < num_batches[i])
            if np.any(use_ix):
                update, cols = cls._compute_update(impact[:, use_ix], target[use_ix])
                probabilities[:, cols] += update
            collect.append(probabilities.copy() if i < num_batches.shape[0] - 1 else probabilities)
            # except for the final iteration, copy the state of probabilities while the original array gets updated in
            # the next iteration
        return [(p, np.sum(p, axis=1)) for p in collect]

    @staticmethod
    def _compute_update(impact, target):
        """Compute update to unscaled probabilities.

        :param impact: as first return value of SetManager._compute_impact(); must be ordered to match target
        :param target: as second return value of SetManager._compute_impact(); must be in ascending order
        :return: two numpy arrays; 2D numpy array of non-negative floats containing increments to be added to unscaled
            probabilities; 1D numpy array of non-negative integers indicating the target values corresponding to the
            columns of the first matrix
        """
        changes = shared.find_changes(target)
        return np.add.reduceat(impact, indices=changes, axis=1), np.unique(target)
