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


MERGE_TOL = 1e-6
# two prototypes are considered identical and suitable for merging if the maximum absolute difference across all feature
# and target values is at most equal to this value
MAX_SAMPLES_FOR_SCORING = 10000  # score no more than this many samples at a time to avoid out-of-memory errors


class SetManager(metaclass=ABCMeta):
    """Abstract base class for set managers.
    """

    _target_type = None  # data type of target depends on subclass

    def __init__(self, target, weights):
        """Initialize set manager.

        :param target: numpy array; target for supervised learning; must have as many elements along the first dimension
            as features has rows
        :param weights: 1D numpy array with non-negative values of type specified by shared.FLOAT_TYPE or None; sample
            weights to be used in the likelihood function; pass None to use unit weights
        """
        self._batches = []
        self._meta = {
            # track properties of the fitting problem that may depend on subclass implementation; this is passed to all
            # static methods called by other public methods than __init__ in case overriding requires additional
            # information
            "num_features": None  # set when adding batches
        }
        # noinspection PyTypeChecker
        self._meta.update(self._get_baseline_distribution(target=target, weights=weights))

    @staticmethod
    @abstractmethod
    def _get_baseline_distribution(target, weights):  # pragma: no cover
        """Compute baseline distribution parameters from target for supervised learning.

        :param target: see docstring of __init__() for details
        :param weights: see docstring of __init__() for details
        :return: dict; contains information regarding the baseline distribution that depends on the model
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
            - prototypes: 2D numpy array of type specified by shared.FLOAT_TYPE; feature matrix of prototypes; sparse
              matrices or infinite/missing values not supported
            - target: 1D numpy array; target for supervised learning; must have as many elements as prototypes has rows
            - feature_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; feature_weights for the batch;
              must have as many elements as prototypes has columns; elements must not be negative
            - prototype_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; prototype weights for the batch;
              must have as many elements as prototypes has rows; elements must not be negative
            - sample_index: 1D numpy integer array; indices of prototypes in training sample; must have as many elements
              as prototypes has rows
        :return: no return values; internal state updated with new batch
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
        shared.check_float_array(x=batch_info["prototypes"], name="prototypes")
        if len(batch_info["target"].shape) != 1:
            raise ValueError("Parameter target must be a 1D array.")
        if batch_info["target"].shape[0] != batch_info["prototypes"].shape[0]:
            raise ValueError("Parameter target must have as many elements as prototypes has rows.")
        if len(batch_info["feature_weights"].shape) != 1:
            raise ValueError("Parameter feature_weights must be a 1D array.")
        if batch_info["feature_weights"].shape[0] != batch_info["prototypes"].shape[1]:
            raise ValueError("Parameter feature_weights must have as many elements as prototypes has columns.")
        shared.check_float_array(x=batch_info["feature_weights"], name="feature_weights")
        if len(batch_info["prototype_weights"].shape) != 1:
            raise ValueError("Parameter prototype_weights must be a 1D array.")
        if batch_info["prototype_weights"].shape[0] != batch_info["prototypes"].shape[0]:
            raise ValueError("Parameter prototype_weights must have as many elements as prototypes has rows.")
        shared.check_float_array(x=batch_info["prototype_weights"], name="prototype_weights")
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
            - scaled_prototypes: 2D numpy array of type specified by shared.FLOAT_TYPE; prototypes reduced to active
              prototypes and features, scaled with feature weights
            - ssq_prototypes: 1D numpy array of type specified by shared.FLOAT_TYPE; row sums of scaled prototypes
            - target: 1D numpy array; target values corresponding to scaled prototypes
            - feature_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; feature weights reduced to active
              features
            - prototype_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; prototype weights reduced to
              active prototypes
            - sample_index: 1D numpy integer array; sample indices reduced to active prototypes
        """
        active_prototypes = np.nonzero(batch_info["prototype_weights"] > 0.0)[0]
        if np.all(active_prototypes.shape[0] == 0):
            return None
        active_features = np.nonzero(batch_info["feature_weights"] > 0.0)[0]
        feature_weights = batch_info["feature_weights"][active_features]
        scaled_prototypes = batch_info["prototypes"][active_prototypes][:, active_features] * feature_weights
        target = batch_info["target"][active_prototypes]
        prototype_weights = batch_info["prototype_weights"][active_prototypes]
        sample_index = batch_info["sample_index"][active_prototypes]
        relation = np.nonzero(pairwise_distances(X=np.hstack([
            scaled_prototypes, target[:, np.newaxis].astype(**shared.FLOAT_TYPE)
        ]), metric="chebyshev") <= MERGE_TOL)
        # find all pairs of features and target that are identical within tolerance
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

        :param features: 2D numpy array of type specified by shared.FLOAT_TYPE; feature matrix for which to compute
            unscaled predictions and scales
        :param num_batches: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or
            None; number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for
            multiple values of num_batches at once
        :return: list of tuples; each tuple consists of two numpy arrays of type specified by shared.FLOAT_TYPE; the
            first array is either 1D or 2D and has the same first dimension as features; it represents the unscaled
            predictions (class probabilities or regression means); the second array is the corresponding scaling vector;
            if an integer is passed for num_batches, the list has length 1; else, the list has one element per element
            of num_batches
        """
        num_batches = self._check_evaluate_input(
            features=features,
            num_batches=num_batches,
            num_batches_actual=self.num_batches,
            permit_array=True,
            meta=self._meta
        )
        if not isinstance(num_batches, np.ndarray):
            num_batches = np.array([num_batches])
        ranges = self._get_sample_ranges(features.shape[0])
        unscaled, scale = self._get_baseline(num_samples=features.shape[0], meta=self._meta)
        result_unscaled = []
        result_scale = []
        if 0 in num_batches:
            result_unscaled.append(unscaled)
            result_scale.append(scale)
        for i in range(num_batches[-1]):  # evaluate up to the largest number of batches requested
            if self._batches[i] is not None:
                new_unscaled = []
                new_scale = []
                for j in range(ranges.shape[0] - 1):
                    range_unscaled, range_scale = self._get_batch_contribution(
                        features=features[ranges[j]:ranges[j + 1], :],
                        batch=self._batches[i],
                        meta=self._meta
                    )
                    new_unscaled.append(range_unscaled)
                    new_scale.append(range_scale)
                unscaled = unscaled + shared.stack_first(new_unscaled)
                scale = scale + np.hstack(new_scale)
            if i + 1 in num_batches:
                result_unscaled.append(unscaled)
                result_scale.append(scale)
        return [(result_unscaled[i], result_scale[i]) for i in range(len(result_unscaled))]

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
        shared.check_float_array(x=features, name="features")
        return cls._check_num_batches(
            num_batches=num_batches, num_batches_actual=num_batches_actual, permit_array=permit_array
        )

    @staticmethod
    def _get_sample_ranges(num_samples):
        """Determine ranges of samples to be scored together.

        :param num_samples: positive integer; number of samples
        :return: 1D numpy integer array; starting points of ranges in increasing order, plus a final value that is one
            greater than the number of samples
        """
        num_ranges = int(np.ceil(num_samples / MAX_SAMPLES_FOR_SCORING))
        return np.hstack([np.arange(num_ranges) * int(np.ceil(num_samples / num_ranges)), num_samples])

    @staticmethod
    @abstractmethod
    def _get_baseline(num_samples, meta):  # pragma: no cover
        """Provide unscaled estimate and scaling for a model with zero batches.

        :param num_samples: positive integer; number of samples
        :param meta: dict; content depends on subclass implementation
        :return: two numpy arrays as a single pair of return values from evaluate_unscaled()
        """
        raise NotImplementedError(
            "Abstract base class SetManager has no default implementation for method _get_baseline()."
        )

    @classmethod
    @abstractmethod
    def _get_batch_contribution(cls, features, batch, meta):  # pragma: no cover
        """Compute contribution of a single batch to the prediction for one set of features.

        :param features: see docstring of evaluate_unscaled() for details
        :param batch: as return value of _process_batch(); None not allowed
        :param meta: dict; content depends on subclass implementation
        :return: two numpy arrays as a single pair of return values from evaluate_unscaled()
        """
        raise NotImplementedError(
            "Abstract base class SetManager has no default implementation for method _get_batch_contribution()."
        )

    def evaluate(self, features, num_batches, compute_familiarity):
        """Compute scaled predictions.

        :param features: 2D numpy array of type specified by shared.FLOAT_TYPE; feature matrix for which to compute
            scaled predictions
        :param num_batches: non-negative integer, 1D numpy array of non-negative and strictly increasing integers, or
            None; number of batches to use for evaluation; pass None for all batches; pass an array to evaluate for
            multiple values of num_batches at once
        :param compute_familiarity: boolean; whether to compute the familiarity for each sample
        :return: one or two lists of numpy arrays of type specified by shared.FLOAT_TYPE; in the first list, each array
            is either 1D or 2D with the same first dimension as features and contains predictions (class probabilities
            or regression means); if an integer is passed for num_batches, the list has length 1; else, the list has one
            element per element of num_batches; the second list is only generated if compute_familiarity = True and
            contains 1D arrays with familiarity scores matching the predictions in the first list
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
            - weight_matrix: 2D numpy array of type specified by shared.FLOAT_TYPE; this has one row per batch and one
              column per feature that is active in at least one batch; features are sorted in order of descending weight
              for the first row, using subsequent rows as tie-breaker
            - feature_index: 1D numpy integer array; index vector indicating the order of features
        """
        num_batches = self._check_num_batches(
            num_batches=num_batches, num_batches_actual=self.num_batches, permit_array=False
        )
        active_features = self.get_active_features(num_batches)
        if active_features.shape[0] == 0:
            return {
                "weight_matrix": np.zeros((num_batches, 0), **shared.FLOAT_TYPE),
                "feature_index": np.zeros(0, dtype=int)
            }
        weight_matrix = []
        for i in range(num_batches):
            new_row = np.zeros(len(active_features), **shared.FLOAT_TYPE)
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

        :param features: 2D numpy array of type specified by shared.FLOAT_TYPE with a single row or None; if not None,
            per-feature similarities are computed between features and each prototype
        :param num_batches: non-negative integer or None; number of batches to export; pass None for all batches
        :return: list whose elements are either dicts or None; None indicates the batch in this position contains no
            prototypes; each dict has the following fields:
            - active_features: 1D numpy integer array; index vector of features with non-zero feature weights
            - prototypes: 2D numpy array of type specified by shared.FLOAT_TYPE; prototypes reduced to active features
            - target: 1D numpy array; target values corresponding to scaled prototypes
            - feature_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; feature weights reduced to active
              features
            - prototype_weights: 1D numpy array of type specified by shared.FLOAT_TYPE; prototype weights
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
            - 1D numpy array of type specified by shared.FLOAT_TYPE or None; features converted to 1D array if not None
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

        :param prototypes: 2D numpy array of type specified by shared.FLOAT_TYPE; prototypes
        :param features: 1D numpy array of type specified by shared.FLOAT_TYPE; features for reference sample
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

    _target_type = {"dtype": int}

    @staticmethod
    def _get_baseline_distribution(target, weights):
        """Compute baseline distribution parameters from target for classification.

        :param target: 1D numpy integer array; class labels encoded as integers from 0 to K - 1
        :param weights: see docstring of SetManager.__init__() for details
        :return: dict with key 'marginals' containing a 1D numpy array of type specified by shared.FLOAT_TYPE with the
            marginal distribution of the classes
        """
        counts = shared.check_classifier_target(target=target, weights=weights)
        return {"marginals": (counts / np.sum(counts))}

    @property
    def marginals(self):
        """Get marginal probabilities.

        :return: 1D numpy array of type specified by shared.FLOAT_TYPE; marginal class probabilities as values in
            (0.0, 1.0)
        """
        # noinspection PyUnresolvedReferences
        return self._meta["marginals"].copy()

    @staticmethod
    def _check_batch(batch_info, meta):
        """Check batch definition for consistent dimensions.

        :param batch_info: see docstring of SetManager._check_batch() for details
        :param meta: dict; must have key 'num_features' but can store None value if not determined yet; must have key
            'marginals' containing a 1D numpy array of type specified by shared.FLOAT_TYPE with the marginal
            distribution of the classes
        :return: as return value of SetManager._check_batch()
        """
        if not np.issubdtype(batch_info["target"].dtype, np.integer):
            raise TypeError("Parameter target must have integer elements.")
        if np.any(batch_info["target"] < 0) or np.any(batch_info["target"] >= meta["marginals"].shape[0]):
            raise ValueError("Parameter target must encode the classes as integers from 0 to K - 1.")
        return SetManager._check_batch(batch_info=batch_info, meta=meta)

    @staticmethod
    def _get_baseline(num_samples, meta):
        """Provide unscaled estimate and scaling for a model with zero batches.

        :param num_samples: positive integer; number of samples
        :param meta: dict; must have key 'marginals' referencing the marginal distribution of classes
        :return: two numpy arrays as a single pair of return values from evaluate_unscaled(); unscaled predictions from
            ClassifierSetManager have a 2D array in first place
        """
        return np.tile(meta["marginals"], (num_samples, 1)).astype(**shared.FLOAT_TYPE), \
            np.ones(num_samples, **shared.FLOAT_TYPE)

    @classmethod
    def _get_batch_contribution(cls, features, batch, meta):
        """Compute contribution of a single batch to the prediction for one set of features.

        :param features: see docstring of evaluate_unscaled() for details
        :param batch: as return value of _process_batch(); None not allowed
        :param meta: dict; must have the following keys:
            - num_features: referencing the expected number of input features
            - marginals: 1D numpy array of floats in [0.0, 1.0); marginal distributions of the classes
        :return: two numpy arrays as a single pair of return values from evaluate_unscaled(); unscaled predictions from
            ClassifierSetManager have a 2D array in first place
        """
        if batch["active_features"].shape[0] == meta["num_features"]:  # no need to reduce input
            scaled_features = features * batch["feature_weights"]
        else:  # reduce input to active features
            scaled_features = features[:, batch["active_features"]] * batch["feature_weights"]
        impact = shared.quick_compute_similarity(
            scaled_reference=scaled_features,
            scaled_prototypes=batch["scaled_prototypes"],
            ssq_reference=np.sum(scaled_features ** 2.0, axis=1),
            ssq_prototypes=batch["ssq_prototypes"]
        ) * batch["prototype_weights"]
        sort_ix = np.argsort(batch["target"])
        # np.add.reduceat() requires prototypes with the same target value grouped together
        changes = shared.find_changes(batch["target"][sort_ix])
        contribution = np.zeros((features.shape[0], meta["marginals"].shape[0]), **shared.FLOAT_TYPE)
        contribution[:, np.unique(batch["target"])] = np.add.reduceat(impact[:, sort_ix], indices=changes, axis=1)
        return contribution, np.sum(contribution, axis=1)
