from IMLearn.metrics import loss_functions
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    df.drop(['id', 'long', 'lat', 'waterfront', 'date'], axis=1, inplace=True)
    df.yr_built = pd.to_datetime(df.yr_built).apply(lambda x: x.value)
    df.yr_renovated = pd.to_datetime(df.yr_renovated).apply(lambda x: x.value)
    df = df[df.price > 0]
    df = df[df.sqft_above >= 0]
    df = df[df.sqft_basement >= 0]
    df = df[df.sqft_lot >= 0]
    df = df[df.sqft_lot15 >= 0]
    df = df[df.sqft_living > 0]
    df = df[df.floors >= 1]
    df = df[(df.bedrooms >= 0) & (df.bedrooms <= 30)]
    df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)
    pricing = df.price
    df.drop(['price'], axis=1, inplace=True)
    return df, pricing


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for x in X.columns:
        temp = X[x]
        c = np.cov(temp, y) / (np.std(temp) * np.std(y))
        c = c.round(3)
        # print(c)
        fig1 = go.Figure([go.Scatter(x=temp, y=y, mode="markers")],
                         # layout=go.Layout(title=r"$\text{" + rf"The Correlation with {x} is " + r"}\begin{bmatrix}" + fr"{c[0][0]} & {c[0][1]}\\{c[1][0]} & {c[1][1]}" + r"\end{bmatrix}$",
                         layout=go.Layout(title=rf"The Correlation of the house pricing with '{x}' is {c[0][1]}",
                                          xaxis_title=f"{x}",
                                          yaxis_title="House Price",
                                          height=1200, width=1200))

        # fig1.show()
        # fig1.write_image(f"../exercises/ex2/{x}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    (data, target) = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, target)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, target)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    thing = LinearRegression()
    all_loss = []
    for p in range(10, 101, 1):
        frac = p / 100
        loss_list = []
        for _ in range(10):
            sample_a = train_X.sample(frac=frac, random_state=_)
            sample_b = train_y.sample(frac=frac, random_state=_)
            thing.fit(sample_a.to_numpy(), sample_b.to_numpy())
            # results = thing.predict(test_X.to_numpy())
            loss_list.append(thing.loss(test_X.to_numpy(), test_y.to_numpy()))
        all_loss.append((p, np.mean(loss_list), np.std(loss_list)))  # (percentage, mean, stdev)
    data = (go.Scatter(x=[x[0] for x in all_loss], y=[x[1] for x in all_loss], mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
             go.Scatter(x=[x[0] for x in all_loss], y=[x[1] - 2 * x[2] for x in all_loss], fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
             go.Scatter(x=[x[0] for x in all_loss], y=[x[1] + 2 * x[2] for x in all_loss], fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False), )
    fig1 = go.Figure(data,
                     layout=go.Layout(title=rf"Loss of results as training data usage increases",
                                      xaxis_title=f"Percentage of training data used",
                                      yaxis_title="Loss",
                                      height=1200, width=1200))

    fig1.show()

    # Quiz
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print(round(loss_functions.mean_square_error(y_true, y_pred), 3))
