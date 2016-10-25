# -*- coding: utf-8 -*-
"""
Created on 2016/10/21 12:11

@author: qiding
"""

# import statsmodels.sandbox.tsa.garch as garch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_generate_process():
    """
    ar(1) - garch(1,1) model
    y_t = a * y_{t-1} + b + a_t
    a_t = \sigma_t * \epsilon_t
    {\sigma_t}^2 = c + d * {a_{t-1}}^2 + e * {\sigma_{t-1}}^2

    :return: y series
    """

    a = 0.8
    b = 0.4
    c = 0.1
    d = 0.3
    e = 0.7
    y_0 = 0.0
    a_0 = 0.2
    sigma_0 = 0.35

    data_len = 10000
    y_series = pd.Series([np.nan] * data_len)
    a_series = pd.Series([np.nan] * data_len)
    sigma_series = pd.Series([np.nan] * data_len)

    epsilon_normal = np.random.normal(loc=0.0, scale=1.0, size=data_len)

    y_series[0] = y_0
    a_series[0] = a_0
    sigma_series[0] = sigma_0

    for idx in range(1, data_len):
        epsilon_t = epsilon_normal[idx]
        sigma_t = np.sqrt(c + d * a_series[idx - 1] ** 2 + e * sigma_series[idx - 1] ** 2)
        a_t = epsilon_t * sigma_t
        y_series[idx] = a * y_series[idx - 1] + b + a_t
        a_series[idx] = a_t
        sigma_series[idx] = sigma_t

    return y_series, a_series, sigma_series


def main():
    y_series, a_series, sigma_series = data_generate_process()
    # y_series = np.exp(1 + y_series)
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)

    ax0.plot(y_series)
    ax1.plot(a_series)
    ax2.plot(sigma_series)

    plt.show()
    pass


if __name__ == '__main__':
    main()
