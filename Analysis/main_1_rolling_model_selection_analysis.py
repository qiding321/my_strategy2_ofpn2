# -*- coding: utf-8 -*-
"""
Created on 2016/11/17 11:51

@author: qiding
"""

from Analysis.func_0_model_selection_functions import *

path_in_root = 'E:\\StrategyResult\\MarketMaking\\' \
               '2016-11-02-12-33-16' \
               'one_reg_jump_hfjumps_model_selection' \
               '__normalize_F_divide_std_F_Logit_truncate_period30_std4_\\'
path_out_root = path_in_root
path_in_file_name = r'accuracy_record.csv'
path_out_file_name = 'deleted_vars_record.txt'

path_in = path_in_root + path_in_file_name
path_out = path_out_root + path_out_file_name


for date_idx in get_dates(path_in):
    all_data, max_acc_data, var_list_del = read_accuracy_file(path_in + date_idx)
