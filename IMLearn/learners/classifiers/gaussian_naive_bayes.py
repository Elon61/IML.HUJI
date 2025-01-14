from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_samples = X.shape[0]  # 300
        # n_features = X.shape[1]  # 2
        self.classes_, indices, counts = np.unique(y, return_counts=True, return_inverse=True)
        self.pi_ = counts / n_samples
        all_mu = []
        all_vars = []

        for i, cls in enumerate(self.classes_):  # for each class, we now have to estimate mu, fig 3.33
            all_mu.append(np.sum(X[indices == i], axis=0) / counts[i])  # TODO does indices contain the correct things?
            all_vars.append(np.sum(np.square(X[indices == i] - all_mu[-1]), axis=0) / counts[i])

        self.mu_ = np.stack(all_mu)  # should be (n_classes,n_features)
        self.vars_ = np.stack(all_vars)  # should be (n_classes,n_features)
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
        """
        likelihoods = self.likelihood(X)
        return self.classes_[np.argmax(likelihoods, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        col = []
        for k in range(len(self.classes_)):
            a = np.log(self.pi_[k] / np.sqrt(2 * np.pi * self.vars_[k]))
            b = np.square(X - self.mu_[k]) / self.vars_[k]
            final = np.sum(a - 0.5 * b, axis=1)
            col.append(final)

            # Alternative, theoretically equivalent, but somehow is not.
            # a = np.log(self.pi_[k] / np.sum(np.sqrt(2 * np.pi * self.vars_[k])))
            # b = []
            # for i in range(X.shape[1]):  # the "i give up" version of this
            #     b.append(0.5 * np.square((X.T[i].T - self.mu_[k][i])) / self.vars_[k][i])
            # b = np.array(b).T
            # final = a - np.sum(b, axis=1)
            # col.append(final)
        return np.array(col).T

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
