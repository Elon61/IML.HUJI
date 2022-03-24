import math

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Quiz:
    arr = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1, -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    gau = UnivariateGaussian()
    gau.fit(arr)
    print("")
    print(np.round(gau.log_likelihood(1, 1, arr), 2))
    print(np.round(gau.log_likelihood(10, 1, arr), 2))

    # Question 1 - Draw samples and print fitted model
    np.random.seed(0)
    samples = np.random.normal(10, 1, 1000)
    gauss = UnivariateGaussian()
    gauss.fit(samples)
    print(f"\n({gauss.mu_}, {gauss.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    many_gauss = [UnivariateGaussian() for _ in range(0, 1000, 10)]
    for i, g in enumerate(many_gauss):
        g.fit(samples[:(i + 1) * 10])
    fitted_data = [(g.mu_, (i + 1) * 10) for i, g in enumerate(many_gauss)]
    fig1 = go.Figure([go.Scatter(x=[y for x, y in fitted_data], y=[abs(x - 10) for x, y in fitted_data], mode='markers+lines', name=r'$\widehat\mu$')],
                     layout=go.Layout(title=r"$\text{Distance of the estimated expectation from the real expectation as function of number of samples}$",
                                      xaxis_title="$m\\text{ - number of samples}$",
                                      yaxis_title="r$|\hat\mu-\mu|$",
                                      height=300, width=1200))
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model]
    PDFs = gauss.pdf(samples)
    fig2 = go.Figure([go.Scatter(x=samples, y=PDFs, mode='markers', name=r'$\widehat\mu$')],
                     layout=go.Layout(title=r"$\text{Empirical PDF of samples}$",
                                      xaxis_title="$Value$",
                                      yaxis_title="r$PDF$",
                                      height=300, width=1200))
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    np.random.seed(0)
    mean = [0, 0, 4, 0]
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean, cov, 1000)
    gauss = MultivariateGaussian()
    gauss.fit(samples)
    print("")
    print(gauss.mu_)
    print(gauss.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    mews = [(np.array([x, 0, y, 0]).T, x, y) for x in f1 for y in f3]
    vals = [(MultivariateGaussian.log_likelihood(mew.T, cov, samples), x, y) for mew, x, y in mews]
    fig1 = go.Figure([go.Heatmap(x=[y for z, x, y in vals], y=[x for z, x, y in vals], z=[z for z, x, y in vals], name=r'$Log-Likelihood$')],
                     layout=go.Layout(title=r"$\text{Log-Likelihood for different input means over the same group of samples}$",
                                      xaxis_title="$f_3$",
                                      yaxis_title="r$f_1$",
                                      height=1200, width=1200))

    fig1.show()

    # Question 6 - Maximum likelihood
    mxval = max(vals, key=lambda x: x[0])
    print(mxval)
    print(f"The maximum likelihood achieved was {np.round(mxval[0], 3)}")
    print(f"For values f1={np.round(mxval[1], 3)}, f3={np.round(mxval[2], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
