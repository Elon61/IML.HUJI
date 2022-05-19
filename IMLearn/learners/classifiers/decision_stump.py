from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is above the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_thresh, best_feature, best_err, sign = 0, 0, 1, 1
        # For every feature
        results = []  # list of (Threshold, Error, Sign, Feature) triplets
        for feature in range(X.shape[1]):
            # Sort the samples
            x_sorted_filtered = X[X[:, feature].argsort()][:, feature]
            y_sorted = y[X[:, feature].argsort()]
            # Find the best threshold, with the positive sign on either side
            results.append((*self._find_threshold(x_sorted_filtered, y_sorted, 1), 1, feature))
            results.append((*self._find_threshold(x_sorted_filtered, y_sorted, -1), -1, feature))

        # Take the split across all features that decides the data best.
        best_thresh, best_err, sign, feature = min(results, key=lambda x: x[1]) # TODO err?

        # at this point, we have a split (threshold value), as well as the label (sign) for anything above the threshold, and the relevant feature (index j).
        # and thus, the single level decision tree is fitted.
        self.threshold_, self.j_, self.sign_ = best_thresh, feature, sign
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self._predict_by_threshold(X[:, self.j_], self.threshold_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        thresh_best, best_err = 0, 1
        # for each pair of consecutive samples
        for i in range(len(values) - 1):
            # split the dataset between the two points
            temp_thresh = np.mean(values[i: i + 2])
            # calculate the accuracy for the split
            temp_pred = self._predict_by_threshold(values, temp_thresh, sign)  # TODO potentially temp_thresh[0] if mean works properly
            temp_err = misclassification_error(temp_pred, labels)
            # update the trackers
            if temp_err < best_err:
                thresh_best = temp_thresh
                best_err = temp_err
        return thresh_best, best_err

    @staticmethod
    def _predict_by_threshold(values: np.ndarray, thresh: float, sign: int):
        return np.sign((values > thresh) - 0.5) * sign

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
