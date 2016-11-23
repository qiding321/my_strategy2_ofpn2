# -*- coding: utf-8 -*-
"""
Created on 2016/11/22 10:28

@author: qiding
"""

import os

import pandas as pd

import log.log
import util.const
from Analysis.func_1_model_selection_modules import OneRegResult


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
    path_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-17-09-29-51rolling_ms_buy_mean_2013080120160731_normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
    reg_type = util.const.FITTING_METHOD.OLS

    my_log = log.log.log_order_flow_predict

    date_path_list = get_path_list(path_root)

    coef_record_all_dates = dict()
    for date_path in date_path_list:
        path_list = get_path_list(date_path)
        this_date = date_path.split('\\')[-2]
        my_log.info('date: {}, length of regressions: {}'.format(this_date, len(path_list)))

        reg_list = []
        for path2 in path_list:
            for path_one_reg in get_path_list(path2):
                this_reg = OneRegResult(result_path=path_one_reg, reg_type=reg_type)
                try:
                    this_reg.update_vars_from_path()
                except FileNotFoundError:
                    my_log.error(path_one_reg)
                    continue
                # print(this_reg)
                reg_list.append(this_reg)

        reg_list_efficient = [one_reg for one_reg in reg_list if one_reg.is_efficient()]
        acc_max = max(one_reg.accuracy for one_reg in reg_list_efficient)
        if reg_type == util.const.FITTING_METHOD.PROBIT or reg_type == util.const.FITTING_METHOD.LOGIT:
            reg_list_max = list(filter(lambda x: acc_max * 0.9 <= x.accuracy <= acc_max, reg_list_efficient))
        else:
            reg_list_max = list(filter(lambda x: acc_max - 0.02 <= x.accuracy <= acc_max, reg_list_efficient))

        longest_var_num = max(len(one_reg.var_list) for one_reg in reg_list_max)
        longest_reg_list = list(filter(lambda x: len(x.var_list) == longest_var_num, reg_list_efficient))

        p_value_record = pd.DataFrame([dict((var_, float(value_)) for var_, value_ in one_reg.p_value.items()) for one_reg in longest_reg_list])
        coef_record = pd.DataFrame([dict((var_, float(value_)) for var_, value_ in one_reg.coef.items()) for one_reg in longest_reg_list])

        # p_value_summary = dict([(k, '\''+str(len(v[v<=0.01]))+'/'+str(len(v.dropna()))) for k, v in p_value_record.iteritems()])
        # coef_summary = dict([(k, '\''+str(len(v[v>=0]))+'/'+str(len(v.dropna()))) for k, v in coef_record.iteritems()])
        p_value_summary = dict([(k, len(v[v<=0.01])/len(v.dropna())) for k, v in p_value_record.iteritems()])
        coef_summary = dict([(k, len(v[v>=0])/len(v.dropna())) for k, v in coef_record.iteritems()])

        coef_summary['accuracy'] = sum([one_reg.accuracy for one_reg in longest_reg_list]) / len(longest_reg_list)

        coef_record_all_dates[this_date] = pd.DataFrame({'p_value': p_value_summary, 'coef_pos': coef_summary}).stack().T

    coef_record_pd = pd.concat(coef_record_all_dates.values(), keys=coef_record_all_dates.keys(), axis=1).T.sort_index()
    # coef_record_pd.to_csv(path_root+'p_value_and_coef_analysis.csv')
    # my_log.info('output_path: {}'.format(path_root+'p_value_and_coef_analysis.csv'))
    coef_record_pd.to_csv(path_root+'p_value_and_coef_analysis_num.csv')
    my_log.info('output_path: {}'.format(path_root+'p_value_and_coef_analysis_num.csv'))


if __name__ == '__main__':
    main()
