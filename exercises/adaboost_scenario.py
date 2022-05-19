import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)

    loss = ada.loss(test_X, test_y)

    print(f"Loss: {loss}")

    fig = go.Figure([go.Scatter(x=[x for x in range(1, n_learners + 1)], y=[ada.partial_loss(train_X, train_y, T) for T in range(1, n_learners + 1)], name="Train Error"),
                     go.Scatter(x=[x for x in range(1, n_learners + 1)], y=[ada.partial_loss(test_X, test_y, T) for T in range(1, n_learners + 1)], name="Test Error")],
                    layout=go.Layout(title=rf"AdaBoost loss as we increase the number of learners, noise={noise}", xaxis_title=f"Learner count", yaxis_title="Loss", height=1200, width=1200))
    fig.update_yaxes(range=[0, max(ada.partial_loss(train_X, train_y, T) for T in range(1, n_learners + 1))])
    fig.update_xaxes(range=[1, n_learners])

    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    symbols = np.array(["circle", "x"])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, m in enumerate(T):
        fig.add_traces([decision_surface(prediction_wrapper(ada, m), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"Decision Boundaries Of AdaBoost on gaussian with noise={noise} and learner count={n_learners}", margin=dict(t=100), height=1300, width=1200) \
        .update_xaxes(visible=True).update_yaxes(visible=True)

    fig.show()  # We can see that due to the use of misclassification loss, the model aims to minimise misclassified examples, and thus at low levels ends up partitioning empty space as that is optimal under that loss,
    # at least we have enough learners.

    # Question 3: Decision surface of best performing ensemble
    best_loss = 1
    best = 0
    for i in T:
        new_loss = ada.partial_loss(test_X, test_y, i)
        print(f"loss for {i} is {new_loss}")
        if new_loss < best_loss:
            best = i
            best_loss = new_loss

    fig = go.Figure([decision_surface(prediction_wrapper(ada, best), lims[0], lims[1], showscale=True),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=True,
                                marker=dict(color=test_y, symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))],
                    layout=go.Layout(title=f"Decision Boundaries Of AdaBoost with {best} learners gaussian with noise={noise} \nand optimal learner count of {n_learners}, achieving an accuracy of {best_loss}", height=1200, width=1200))
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure([decision_surface(ada.predict, lims[0], lims[1], showscale=True),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=True, marker_size=ada.D_ * len(ada.D_),
                                marker=dict(color=test_y, symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]], #sizemode='area', sizeref=1, size=ada.D_,
                                            line=dict(color="black", width=1)))],
                    layout=go.Layout(title=f"Decision Boundaries Of AdaBoost with {best} learners gaussian with noise={noise} and n_learners={n_learners}, \nwith samples displayed by their final weights", height=1200, width=1200))
    fig.show()


def prediction_wrapper(model, T):
    def _inner(X):
        return model.partial_predict(X, T)

    return _inner


if __name__ == '__main__':
    np.random.seed(0)
    # fit_and_evaluate_adaboost(0, 250)
    fit_and_evaluate_adaboost(0.4, 250)
