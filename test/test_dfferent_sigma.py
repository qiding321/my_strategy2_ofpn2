# -*- coding: utf-8 -*-
"""
Created on 2016/10/23 13:56

@author: qiding
"""

import matplotlib.pyplot as plt
import numpy as np


def data_generate_process():
    data_len = 1000
    data = np.array([np.nan] * data_len)
    sigma_max = 1

    sigma = np.arange(0, sigma_max, sigma_max / data_len)
    # sigma = np.array([sigma_max]*data_len)
    mu = 0.5

    dt = 1 / data_len
    dB = np.random.normal(0, 1 * np.sqrt(dt), data_len)

    data[0] = 1
    dp = 0
    for idx_ in range(1, data_len):
        sigma_t = sigma[idx_]
        dB_t = dB[idx_]
        dp = (-dp * dt + sigma_t * dB_t) * data[idx_ - 1]
        # dp = (mu * dt + sigma_t * dB_t)
        # data[idx_] = dp
        data[idx_] = data[idx_ - 1] + dp

    return data


def main():
    # data0 = np.array([np.nan]*1000)
    # for i in range(1000):
    #     data = data_generate_process()
    #     data0[i] = data[-1]
    # plt.hist(data0, 100)
    # mean = data0.mean()
    # std = data0.std()
    # plt.title('mean: {:.2f}, std: {:.2f}'.format(mean, std))
    # plt.show()

    data = data_generate_process()
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    main()
