from sklearn.model_selection import train_test_split

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str, dummy=True) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df = df[df.Temp >= -70]  # units are unclear, but still
    df = df[df.Year > 1800]
    df = df[df.Day != 0]
    # df.drop(['Day', 'Year', 'Month'], axis=1, inplace=True)
    df.drop(['Day'], axis=1, inplace=True)
    df['DayOfYear'] = df.Date.dt.day_of_year
    df.drop(['Date'], axis=1, inplace=True)

    if dummy:
        df = pd.get_dummies(df, columns=['Country', 'City'])  # , drop_first=True)
    # target = df.Temp
    # df.drop(['Temp'], axis=1, inplace=True)
    return df  # , target


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data.Country_Israel == 1].drop([x for x in data.columns if "City" in x or "Country" in x], axis=1)
    israel_data.Year = israel_data.Year.astype(str)
    fig1 = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                      title=f"Relationship between day of the year, temperature, and year.", height=1200, width=1200)
    # fig1.show()  # degree two, or perhaps three

    stdev = israel_data.groupby("Month").std()
    fig2 = px.bar(stdev, x=stdev.index, y="Temp",
                  title=f"Standard deviation of the average temperature for each month.", height=1200, width=1200)
    # fig2.show()

    # Question 3 - Exploring differences between countries
    data = load_data("../datasets/City_Temperature.csv", False).drop(["Year", "DayOfYear"], axis=1)

    stdev_3 = data.groupby(["Country", "Month"]).std()
    mean_3 = data.groupby(["Country", "Month"]).mean()
    data3 = stdev_3.join(mean_3, rsuffix="_mean", lsuffix="_stdev")
    plot_data = []
    for country in data3.index.unique("Country"):
        curr_data = data3.loc[country]
        # plot_data.append((go.Scatter(x=curr_data.index, y=curr_data.Temp_mean, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
        #                   go.Scatter(x=curr_data.index, y=curr_data.Temp_mean + 2 * curr_data.Temp_stdev, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
        #                   go.Scatter(x=curr_data.index, y=curr_data.Temp_mean + 2 * curr_data.Temp_stdev, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False), ))

        plot_data.extend((go.Scatter(x=curr_data.index, y=curr_data.Temp_mean, mode="markers+lines", name=f"{country}"),
                          go.Scatter(x=curr_data.index, y=np.asarray(curr_data.Temp_mean + 2 * curr_data.Temp_stdev), fill=None, mode="lines", showlegend=False, name=f"stdev_{country}"),
                          go.Scatter(x=curr_data.index, y=np.asarray(curr_data.Temp_mean - 2 * curr_data.Temp_stdev), fill='tonexty', mode="lines", showlegend=False, name=f"stdev_{country}")))
    fig3 = go.Figure(plot_data,
                     layout=go.Layout(title=rf"Average monthly temperature, along with stdev, per country",
                                      xaxis_title=f"Month",
                                      yaxis_title="Temperature",
                                      height=1200, width=1200))

    # fig3.show()

    # Question 4 - Fitting model for different values of `k`
    losses = []
    train_data, train_labels, test_data, test_labels = split_train_test(israel_data.DayOfYear, israel_data.Temp)
    for k in range(1, 11):
        pl = PolynomialFitting(k)
        pl.fit(np.array(train_data), np.array(train_labels))
        loss = np.log(pl.loss(np.array(test_data), np.array(test_labels)))
        print(loss)
        losses.append(loss)

    fig4 = go.Figure(go.Bar(x=[x for x in range(1, 11)], y=losses, ),
                     layout=go.Layout(title=rf"Log(Loss) as a function of K", xaxis_title=f"K", yaxis_title="Loss", height=1200, width=1200))
    fig4.update_yaxes(type="log")

    # fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    data = load_data("../datasets/City_Temperature.csv")
    print(data.columns)
    jordan_data = data[data.Country_Jordan == 1].drop([x for x in data.columns if "City" in x or "Country" in x], axis=1)
    nether_data = data[data["Country_South Africa"] == 1].drop([x for x in data.columns if "City" in x or "Country" in x], axis=1)
    south_data = data[data["Country_The Netherlands"] == 1].drop([x for x in data.columns if "City" in x or "Country" in x], axis=1)

    test_jordan_data, test_jordan_labels, _, _ = split_train_test(jordan_data.DayOfYear, jordan_data.Temp, 1)
    test_nether_data, test_nether_labels, _, _ = split_train_test(nether_data.DayOfYear, nether_data.Temp, 1)
    test_south_data, test_south_labels, _, _ = split_train_test(south_data.DayOfYear, south_data.Temp, 1)

    pl = PolynomialFitting(3)
    pl.fit(np.array(train_data), np.array(train_labels))

    losses = [(pl.loss(np.array(test_data), np.array(test_labels))),
              (pl.loss(np.array(test_jordan_data), np.array(test_jordan_labels))),
              (pl.loss(np.array(test_nether_data), np.array(test_nether_labels))),
              (pl.loss(np.array(test_south_data), np.array(test_south_labels)))]  # israel, jordan, netherlands, south africa
    fig5 = go.Figure(go.Bar(x=["Israel", "Jordan", "Netherlands", "South Africa"], y=losses),
                     layout=go.Layout(title=rf"Loss of the model on the different countries in the dataset", xaxis_title=f"Country", yaxis_title="Loss", height=1200, width=1200))

    # fig5.show()
