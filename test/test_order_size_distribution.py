# -*- coding: utf-8 -*-
"""
Created on 2016/10/18 10:07

@author: qiding
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import my_path.path


def main():
    path_file_in_root = my_path.path.raw_data_source_path + 'OrderQueue\\SH\\'
    path_out = my_path.path.market_making_result_root + 'order_size_distribution\\'

    date_list = os.listdir(path_file_in_root)

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    for num, date in enumerate(date_list):
        print(num, len(date_list))
        file_path = path_file_in_root + date + '\\' + '601818.csv'
        try:
            data = pd.read_csv(file_path, encoding='gbk')
        except OSError:
            continue
        column_ab = [col_name for col_name in data.columns if col_name.startswith('ab')]
        data_ab = pd.DataFrame(data[column_ab] / 100)
        data_list = list(data_ab.stack())
        data_list_filtered = pd.Series(list(filter(lambda x: x >= 0, data_list)))

        mean = np.mean(data_list_filtered)
        std = np.std(data_list_filtered)
        kurt = data_list_filtered.kurt()
        skew = data_list_filtered.skew()
        title = 'mean: {:.1f}, std: {:.1f}, kurt: {:.1f}, skew: {:.1f}'.format(mean, std, kurt, skew)

        plt.hist(data_list, 100, log=True)
        plt.xlim([0, 10000])
        plt.title(title)
        for av_num in range(1, 10):
            plt.axvline(av_num * 1000, color='r')
        plt.savefig(path_out + date + '.jpg')
        plt.close()


if __name__ == '__main__':
    main()
