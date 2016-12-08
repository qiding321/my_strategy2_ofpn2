# -*- coding: utf-8 -*-
"""
Created on 2016/12/7 11:22

@version: python3.5
@author: qiding
"""

import os

import pandas as pd

import util.const


def get_path_list(path_root):
    path_list = []
    for folder1 in os.listdir(path_root):
        if not os.path.isdir(path_root + folder1):
            continue
        path_list.append(path_root+folder1+'\\')
    return path_list


def main(input_file_name, output_file_name, percent, record_type):

    ########################## config ###########################################################
    reg_type = util.const.FITTING_METHOD.LOGIT
    path_root = r'\\SAS5\Users\dqi\Documents\Output\MarketMaking' + '\\'
    folders_list = [
        '2016-12-06-18-28-31sell_jump_selected_rolling_cutoffand3month__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-28-23sell_jump_cy_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-07-10-39-51sell_jump_yy_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-28-40sell_jump_selected_rolling_cutoffand3month_strict__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-28-03sell_jump_qd_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-27-46sell_jump_selected_rolling__normalize_F_divide_std_F_Logit_truncate_period30_std4_',

        '2016-12-06-18-09-13buy_jump_selected_rolling_cutoffand3month_strict__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-08-39buy_jump_selected_rolling__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-11-14buy_jump_qd_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-24-26buy_jump_cy_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-06-18-08-58buy_jump_selected_rolling_cutoffand3month__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
        '2016-12-07-10-40-25buy_jump_yy_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_',
    ]
    labels_list = [
        'sell_3month',
        'sell_manually_cy',
        'sell_manually_yy',
        'sell_3month_strict',
        'sell_manually_qd',
        'sell_onemonth',

        'buy_3month_strict',
        'buy_onemonth',
        'buy_manually_qd',
        'buy_manually_cy',
        'buy_3month',
        'buy_manually_yy',
    ]
    output_path = r'E:\StrategyResult\MarketMaking\model selection result' + '\\' + \
                  output_file_name

    ########################## data ###########################################################
    accuracy_list = []
    for this_folder in folders_list:
        this_path = path_root + this_folder + '\\'
        date_path_list = get_path_list(this_path)
        print('\n'.join(date_path_list))

        record_list = {}

        for date_path in date_path_list:
            this_date = date_path.split('\\')[-2]
            if reg_type == util.const.FITTING_METHOD.LOGIT:
                file_path = date_path + input_file_name
                data_ = pd.read_csv(file_path)
                if record_type == 'accuracy':
                    if percent is not None:
                        data_tocal = data_[data_['pct_low']>=percent]
                        max_value = data_tocal['target_num'].sum()/data_tocal['all_num'].sum() if data_tocal['all_num'].sum() >= 30 else 0
                    else:
                        max_value = data_['accuracy'].iloc[-1] if data_['all_num'].iloc[-1] >= 30 else 0
                elif record_type == 'all_num':
                    if percent is not None:
                        data_tocal = data_[data_['pct_low']>=percent]
                        max_value = data_tocal['all_num'].sum()
                    else:
                        max_value = data_['all_num'].iloc[-1]
                elif record_type == 'target_num':
                    if percent is not None:
                        data_tocal = data_[data_['pct_low']>=percent]
                        max_value = data_tocal['target_num'].sum()
                    else:
                        max_value = data_['target_num'].iloc[-1]
                else:
                    raise ValueError

            else:
                file_path = date_path + 'daily_rsquared.csv'
                data_ = pd.read_csv(file_path)
                msr = data_['msr'].sum()
                mse = data_['mse'].sum()
                max_value = 1 - msr / mse
            record_list[this_date] = max_value
        record_df = pd.Series(record_list).sort_index()
        print('generate csv: {}'.format(this_path+'accuracy_record.csv'))
        record_df.to_csv(this_path+'accuracy_record.csv')
        accuracy_list.append(record_df)

    ########################## unconditional ###########################################################
    if reg_type == util.const.FITTING_METHOD.LOGIT:
        # sell
        this_folder = '2016-12-06-18-28-31sell_jump_selected_rolling_cutoffand3month__normalize_F_divide_std_F_Logit_truncate_period30_std4_'
        this_path = path_root + this_folder + '\\'
        date_path_list = get_path_list(this_path)
        print('\n'.join(date_path_list))

        record_list = {}

        for date_path in date_path_list:
            this_date = date_path.split('\\')[-2]
            file_path = date_path + input_file_name
            data_ = pd.read_csv(file_path)
            max_value = data_['target_num'].sum() / data_['all_num'].sum()
            record_list[this_date] = max_value
        record_df = pd.Series(record_list).sort_index()
        accuracy_list.append(record_df)
        labels_list.append('sell_acc_unconditional')

        # buy
        this_folder = '2016-12-06-18-09-13buy_jump_selected_rolling_cutoffand3month_strict__normalize_F_divide_std_F_Logit_truncate_period30_std4_'
        this_path = path_root + this_folder + '\\'
        date_path_list = get_path_list(this_path)
        print('\n'.join(date_path_list))

        record_list = {}

        for date_path in date_path_list:
            this_date = date_path.split('\\')[-2]
            file_path = date_path + input_file_name
            data_ = pd.read_csv(file_path)
            max_value = data_['target_num'].sum() / data_['all_num'].sum()
            record_list[this_date] = max_value
        record_df = pd.Series(record_list).sort_index()
        accuracy_list.append(record_df)
        labels_list.append('buy_acc_unconditional')

    ########################## merge ###########################################################

    record_pd = pd.DataFrame(pd.concat(accuracy_list, axis=1, keys=labels_list))
    record_pd.to_csv(output_path)



if __name__ == '__main__':
    input_file_name_list = [
        'var_analysis\\out_of_sample_cutoffpercentile.csv',
        'var_analysis\\out_of_sample_cutoffpercentile.csv',
        'var_analysis\\out_of_sample10percentile.csv',
        'var_analysis\\out_of_sample40percentile.csv',
        ]

    output_file_name_list = [
        'compare_20-100pct.csv',
        'compare_80-100pct.csv',
        'compare_10percentile.csv',
        'compare_40percentile.csv',
        ]

    percent_list = [
        0.2,
        0.8,
        None,
        None
    ]

    record_type = 'target_num'
    # record_type = 'all_num'
    # record_type = 'accuracy'

    for input_file_name, output_file_name, percent in zip(input_file_name_list, output_file_name_list, percent_list):
        main(input_file_name, record_type+'_'+output_file_name, percent=percent, record_type=record_type)
