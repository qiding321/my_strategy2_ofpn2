# -*- coding: utf-8 -*-
"""
Created on 2016/11/9 19:18

@author: qiding
"""

# path_in_root ='E:\\StrategyResult\\MarketMaking\\2016-11-11-10-28-12sell_mean_ms__normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
# path_in_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-11-10-27-47buy_mean_ms__normalize_F_divide_std_F_OLS_truncate_period30_std4_\\'
# path_in_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-08-15-55-54one_reg_sell_jump_ms_model_selection__normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
path_in_root = 'E:\\StrategyResult\\MarketMaking\\2016-11-02-12-33-16one_reg_jump_hfjumps_model_selection__normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
path_out_root = path_in_root
path_in_file_name = r'accuracy_record.csv'
path_out_file_name = 'deleted_vars_record.txt'

path_in = path_in_root + path_in_file_name
path_out = path_out_root + path_out_file_name

lines = []

with open(path_in, 'r') as f_in:
    for line in f_in.readlines():
        lines.append(line)

num = []
acc = []
vars_ = []

for line in lines[1:]:
    data_tmp = line.split(',')
    num.append(int(data_tmp[0]))
    acc.append(float(data_tmp[1]))
    vars_.append(data_tmp[4:])

var_list = []

for num_ in range(min(num), max(num) + 1):
    idx_ = []
    acc_ = []
    for idx_tmp, n_ in enumerate(num):
        if n_ == num_:
            idx_.append(idx_tmp)
            acc_.append(acc[idx_tmp])
    acc_max = max(acc_)
    for idx_tmp, acc_tmp in enumerate(acc_):
        if acc_tmp == acc_max:
            idx_max = idx_[idx_tmp]
            break
    else:
        raise ValueError

    var_left = vars_[idx_max]
    var_list.append(var_left)

var_add_list = []
for var_1, var_2 in zip(var_list[0:-1], var_list[1:]):
    var_1_ = [x.replace('\n', '') for x in var_1]
    var_2_ = [x.replace('\n', '') for x in var_2]

    var_add = [x for x in var_2_ if x not in var_1_]
    var_add_list.append(var_add)

with open(path_out, 'w') as f_out:
    for var_l in var_add_list[::-1]:
        f_out.write(var_l[0].replace('_x', '') + '\n')
