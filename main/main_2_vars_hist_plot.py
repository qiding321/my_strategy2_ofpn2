# -*- coding: utf-8 -*-
"""
Created on 2016/10/12 16:59

@author: qiding
"""


import pickle as pkl
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import data.data
import data.reg_data
import log.log
import my_path.path
import util.util
from paras.paras import Paras


my_log = log.log.log_order_flow_predict


def main():
    # ==========================parameters and path======================
    my_para = Paras()
    output_path = my_path.path.market_making_result_root + my_para.get_title() + '\\'

    # =========================log================================
    my_log.add_path(log_path2=output_path + 'log.log')
    my_log.info('paras:\n%s' % my_para)
    my_log.info('output path:\n{}'.format(output_path))

    # ============================loading data from csv====================
    my_log.info('data begin')

    # data_training = data.data.TrainingData(this_paras=my_para)

    # util.util.dump_pkl(data_training, my_path.path.unit_test_data_path + 'data_training.pkl')
    data_training = util.util.load_pkl(my_path.path.unit_test_data_path + 'data_training.pkl')
    data_training.paras = my_para

    my_log.info('data end')
    # assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)

    # ============================reg data=================
    reg_data_training, normalize_funcs = data_training.generate_reg_data()
    # reg_data_testing, _ = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs, reg_data_training=reg_data_training)

    assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
    # assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

    vars_to_hist = [
        # 'buyvolume_x', 'sellvolume_x',
        'bsize1_change_x', 'asize2_x',
        ]

    data_type_list = [
        'raw',
        'truncate2', 'truncate3', 'truncate4', 'truncate5',
        'truncate_tail2', 'truncate_tail3', 'truncate_tail4', 'truncate_tail5',
        'log', 'zscore', 'zscore_log']
    # data_type_list = ['zscore_log']

    for var_name in vars_to_hist:
        output_path_ = output_path + var_name + '\\'
        if os.path.exists(output_path_):
            pass
        else:
            os.makedirs(output_path_)
            my_log.info('make dirs: {}'.format(output_path_))
        var_ = reg_data_training.x_vars[var_name]
        for data_type in data_type_list:
            hist_this_var(var_, data_type, output_path_, var_name)


def hist_this_var(var_raw, data_type, output_path_, var_name):
    figure_path = output_path_ + var_name + data_type + '.jpg'
    csv_path = output_path_ + var_name + data_type + '.csv'

    my_log.info(var_name + data_type + ' begin')
    var_, _ = util.util.winsorize(var_raw, [0.01, 0.99])
    if data_type == 'raw':
        var_to_hist = var_
    elif data_type.startswith('truncate') and not data_type.startswith('truncate_tail'):
        truncate_std = int(data_type[-1])
        var_truncated, var_tail = get_truncated(var_col=var_, window=30, truncate_std=truncate_std)
        var_to_hist = var_truncated
    elif data_type.startswith('truncate_tail'):
        truncate_std = int(data_type[-1])
        var_truncated, var_tail = get_truncated(var_col=var_, window=30, truncate_std=truncate_std)
        var_to_hist = var_tail
    elif data_type == 'zscore':
        var_to_hist = get_zcore(var_col=var_, window=30)
    elif data_type == 'zscore_log':
        z_score = get_zcore(var_col=var_, window=30)
        var_to_hist = pd.concat([np.log(z_score[z_score>0]), np.log(-z_score[z_score<0])])
    elif data_type == 'log':
        var_to_hist = take_log_and_truncate(var_col=var_)
    else:
        raise LookupError
    my_log.info(var_name + data_type + ' data gen complete')

    plt.hist(var_to_hist.dropna(), 100, color='r')
    plt.savefig(figure_path)
    plt.close()
    my_log.info(var_name + data_type + ' plot complete @{}'.format(figure_path))

    var_description(var_to_hist.dropna(), csv_path)
    my_log.info(var_name + data_type + ' csv complete @{}'.format(csv_path))


def get_zcore(var_col, window):
        n_ = len(var_col)
        var_col_new = pd.Series([np.nan] * n_, index=var_col.index)
        var_col_new.iloc[0:window] = var_col.iloc[0:window]
        for i in range(window, n_):
            var_tmp = var_col_new[i - window:i]
            mean_tmp = var_tmp.mean()
            std_tmp = var_tmp.std()
            point_raw = var_col.iloc[i]
            point_new = (point_raw - mean_tmp) / std_tmp
            var_col_new.iloc[i] = point_new
        return var_col_new


def get_truncated(var_col, window, truncate_std):
    n_ = len(var_col)
    var_col_new = pd.Series([np.nan] * n_, index=var_col.index)
    truncate_tail = pd.Series([np.nan] * n_, index=var_col.index)
    var_col_new.iloc[0:window] = var_col.iloc[0:window]
    for i in range(window, n_):
        var_tmp = var_col_new[i - window:i]
        mean_tmp = var_tmp.mean()
        std_tmp = var_tmp.std()
        point_raw = var_col.iloc[i]
        if point_raw >= mean_tmp + truncate_std * std_tmp:
            point_new = mean_tmp + truncate_std * std_tmp
            point_tail = point_raw
        else:
            point_new = point_raw
            point_tail = np.nan
        var_col_new.iloc[i] = point_new
        truncate_tail.iloc[i] = point_tail

    return var_col_new, truncate_tail


def take_log_and_truncate(var_col):
    zero_index = var_col <= 0
    var_col[zero_index] = np.nan
    min_ = var_col.min()
    var_col[zero_index] = min_
    var_col_log = np.log(var_col)
    return var_col_log


def var_description(series, output_path):
    err_des = series.describe()
    err_des['skew'] = series.skew()
    err_des['kurt'] = series.kurt()

    err_des.to_csv(output_path)

if __name__ == '__main__':
    main()
















