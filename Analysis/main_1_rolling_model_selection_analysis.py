# -*- coding: utf-8 -*-
"""
Created on 2016/11/17 11:51

@author: qiding
"""

from collections import Counter

from Analysis.func_0_model_selection_functions import *

path_in_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-16-15-27-53rolling_ms_2014030120160731_normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
path_out_root = path_in_root
path_in_file_name = r'accuracy_record.csv'
path_out_file_name = 'accuracy_analysis.txt'

path_out = path_out_root + path_out_file_name
with open(path_out, 'w') as f_out:
    f_out.write('')

date_list = get_dates(path_in_root)

var_list_in_all_all_date = []

for date_idx in date_list:
    all_data, max_acc_data, var_list_del = read_accuracy_file(path_in_root + date_idx + '\\' + path_in_file_name)

    max_acc_list = max_acc_data['acc_list_max']
    var_list_max = max_acc_data['var_list_max']
    hit_max_list = max_acc_data['hit_list_max']
    all_num_max_list = max_acc_data['all_num_list_max']

    max_acc = max(max_acc_list)
    max_acc_down_limit = max_acc * 0.9
    idx_list_max_range_acc = [idx_ for idx_ in range(len(max_acc_list)) if max_acc_down_limit <= max_acc_list[idx_] <= max_acc]

    var_list_max_range = [var_list_max[idx_] for idx_ in idx_list_max_range_acc]
    acc_list_max_range = [max_acc_list[idx_] for idx_ in idx_list_max_range_acc]
    hit_max_range = [hit_max_list[idx_] for idx_ in idx_list_max_range_acc]
    all_num_max_range = [all_num_max_list[idx_] for idx_ in idx_list_max_range_acc]

    with open(path_out, 'a') as f_out:
        f_out.write('date: {}\n\t'.format(date_idx))
        f_out.write(
            '\n\t'.join(
                (str(acc_) + ': ' + str(len(var_)) + ', ' + str(hit_) + '/' + str(all_num_) + ', ' + ', '.join(var_)
                 for acc_, var_, hit_, all_num_ in zip(acc_list_max_range, var_list_max_range, hit_max_range, all_num_max_range))
            )
        )
        f_out.write('\n')

    var_list_in_all_this_date_set = set()
    for var_list_ in var_list_max_range:
        var_list_in_all_this_date_set = var_list_in_all_this_date_set.union(var_list_)
    var_list_in_all_this_date = list(var_list_in_all_this_date_set)
    var_list_in_all_all_date.append(var_list_in_all_this_date)

var_list_all = []
[var_list_all.append(var_) for var_list__ in var_list_in_all_all_date for var_ in var_list__]

var_count = Counter(var_list_all)
print(var_count)
