import numpy as np
from base import BaseEstimator
from typing import Callable, NoReturn

from metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        rng = np.random.default_rng()
        data_ = np.c_[X, y]  # TODO does c_ work?
        # weights stuff (D)
        d = self.normalize(np.ones(X.shape[0]))  # sample weights
        models = []
        weights = []
        # iterate for self.iterations_ times.
        for i in range(self.iterations_):
            if i % 10 == 0:
                print(f"Info: round {i} / {self.iterations_}")
            # Draw a new set of samples based on the new weights
            sampled = rng.choice(data_, len(X), p=d, axis=0)
            sampleX = sampled[:, :X.shape[1]]  # TODO does this work?
            sampley = sampled[:, X.shape[1]:]
            # fit a decision stump over the data
            model = self.wl_()
            model.fit(sampleX, sampley)
            pred = model.predict(X)
            # using the returned parameters, updates the weights (D),
            w, d = self.weight_calc(d, y, pred)
            # store the model
            models.append(model)
            weights.append(w)
        self.models_, self.weights_, self.D_ = np.array(models), np.array(weights), np.array(d)
        # the estimator is now fitted.
        self.fitted_ = True

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        return v / norm

    def weight_calc(self, d, y, h):
        """
        :param d: previous weight vector
        :param y: real labels
        :param h: predicted labels
        :return:
        """
        eps = np.sum(d[y != h])  # Epsilon is the sum of weights where the predictor is wrong
        w = 0.5 * np.log((1 / eps) - 1)  # exponent vector calculated as per the formula
        d = d * np.exp(-w * y * h)  # new weights
        return w, self.normalize(d)  # normalise the weights and return

    def _predict(self, X):
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
        """
        return self.partial_predict(X, self.iterations_)

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if T > self.iterations_ or T < 1:
            raise ValueError("Value T out of range")
        h = np.array([self.models_[i].predict(X) for i in range(T)])
        weights = self.weights_[:T, None]
        return np.sign(np.sum(h * weights, axis=0))

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
