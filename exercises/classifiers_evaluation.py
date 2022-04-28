from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(f"../datasets/{f}")
        split_data = np.hsplit(data, (data.shape[1] - 1, data.shape[1]))
        features = split_data[0]
        labels = split_data[1]

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_func(perceptron, sample, response):
            losses.append((len(losses), perceptron.loss(features, labels)))

        model = Perceptron(callback=callback_func)
        model.fit(features, labels)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(go.Scatter(x=[x for x, y in losses], y=[y for x, y in losses]),
                        layout=go.Layout(title=rf"Loss of the model at each iteration on the {n} dataset", xaxis_title=f"Iteration", yaxis_title="Loss", height=1200, width=1200))
        fig.update_yaxes(range=[0, 1])

        # fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load(f"../datasets/{f}")
        split_data = np.hsplit(data, (data.shape[1] - 1, data.shape[1]))
        features = split_data[0]
        labels = split_data[1]

        # Fit models and predict over training set
        model_LDA = LDA()
        model_LDA.fit(features, labels)
        model_naive_bayes = GaussianNaiveBayes()
        model_naive_bayes.fit(features, labels)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        acc_LDA = accuracy(labels, model_LDA.predict(features))
        acc_bayes = accuracy(labels, model_naive_bayes.predict(features))

        fig = make_subplots(1, 2, subplot_titles=(f"LDA: Acc={acc_LDA}", f"Bayes: Acc={acc_bayes}"))


        # Add traces for data-points setting symbols and colors
        fig1 = generate_fig(f, features, model_LDA.predict(features), acc_LDA, "LDA")
        fig2 = generate_fig(f, features, model_naive_bayes.predict(features), acc_bayes, "Bayes")
        for trace in fig1.data:
            fig.append_trace(trace, 1, 1)
        fig.update_layout()
        for trace in fig2.data:
            fig.append_trace(trace, 1, 2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.append_trace(go.Scatter(x=[x for x, y in model_LDA.mu_], y=[y for x, y in model_LDA.mu_], mode="markers", marker={"symbol": "x-dot", "color": "black"}, name="Mean LDA"), 1, 1)
        fig.append_trace(go.Scatter(x=[x for x, y in model_naive_bayes.mu_], y=[y for x, y in model_naive_bayes.mu_], mode="markers", marker={"symbol": "x-dot", "color": "black"}, name="Mean Bayes"), 1, 2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i, (x, y) in enumerate(model_LDA.mu_):
            fig.add_trace(get_ellipse(model_LDA.mu_[i], model_LDA.cov_))
        for i, (x, y) in enumerate(model_naive_bayes.mu_):
            x1, y1 = model_naive_bayes.vars_[i]
            fig.add_shape(type="circle", xref="x", yref="y", x0=x-x1, y0=y-y1, x1=x+x1, y1=y+y1, opacity=0.5, row=1, col=2, fillcolor="White", line=dict(color="Black", width=3), layer="below")

        fig.update_layout(height=1200, width=2400, title_text=f"Group predictions on the dataset {f}", title_x=0.5)
        fig.show()


def generate_fig(dataset_name, features, labels, accuracy, model_name, height=1200, width=1200):
    unique_labels = np.unique(labels)
    plots = [go.Scatter(x=[x for i, (x, y) in enumerate(features) if labels[i] == _x],
                        y=[y for i, (x, y) in enumerate(features) if labels[i] == _x], mode="markers", name=f"Group {_x}") for _i, _x in enumerate(unique_labels)]
    fig = go.Figure(plots, layout=go.Layout(title=f"Group predictions using {model_name} on the {dataset_name} dataset, the accuracy is {accuracy}",
                                            xaxis_title=f"x", yaxis_title="x", height=height, width=width))
    return fig


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
