# -*- coding: utf-8 -*-
"""
Created on 2016/11/29 15:58

@author: qiding
"""

import os

import pandas as pd

import log.log
import util.const


def get_path_list(path_root):
    path_list = []
    for folder1 in os.listdir(path_root):
        if not os.path.isdir(path_root + folder1):
            continue
        path_list.append(path_root+folder1+'\\')
    return path_list


def main():
    # path_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-16-15-27-53rolling_ms_2014030120160731_normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.LOGIT
    # path_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-17-09-43-25rolling_ms_sell_mean_2013080120160731_normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.OLS
    # path_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-16-09-13-21rolling_ms_2013080120160731_normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.LOGIT
    # path_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-17-09-29-51rolling_ms_buy_mean_2013080120160731_normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.OLS
    # path_root = r'E:\StrategyResult\MarketMaking\2016-11-24-16-22-00rolling_ms_buy_mean_2013080120160731_normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.OLS
    # path_root = r'E:\StrategyResult\MarketMaking\2016-11-24-16-21-44rolling_ms_sell_mean_2013080120160731_normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.OLS
    # path_root = r'E:\StrategyResult\MarketMaking\2016-11-28-09-51-59rolling_ms_buy_jump_10bins_2013080120160731_normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
    # reg_type = util.const.FITTING_METHOD.LOGIT
    path_root = r'E:\StrategyResult\MarketMaking\2016-11-28-09-52-27rolling_ms_sell_jump_10bins_2013080120160731_normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
    reg_type = util.const.FITTING_METHOD.LOGIT

    my_log = log.log.log_order_flow_predict

    date_path_list = get_path_list(path_root)


    record_list = {}

    for date_path in date_path_list:
        this_date = date_path.split('\\')[-2]
        file_path = date_path + 'accuracy_record.csv'

        max_bar_acc_this_month = []
        with open(file_path, 'r') as f_in:
            first_flag = True
            for line in f_in.readlines():
                if first_flag:
                    first_flag = False
                else:
                    words = line.split(',')
                    max_bar_acc_ = float(words[1])
                    max_bar_len_ = float(words[3])
                    if reg_type == util.const.FITTING_METHOD.LOGIT:
                        if max_bar_len_ >= 100:
                            max_bar_acc_this_month.append(max_bar_acc_)
                    else:
                        max_bar_acc_this_month.append(max_bar_acc_)
        max_value = max(max_bar_acc_this_month)
        record_list[this_date] = max_value
    record_df = pd.Series(record_list)
    record_df.to_csv(path_root+'max_accuracy.csv')


if __name__ == '__main__':
    main()
