# -*- coding: utf-8 -*-
"""
Created on 2016/11/17 11:59

@author: qiding
"""

import os


def read_accuracy_file(file_path):
    lines_list = _read_raw_txt_file(file_path)

    num_list, acc_list, vars_list, hit_list, all_num_list = _get_all_data(lines_list)

    num_list_max, acc_list_max, var_list_max, hit_list_max, all_num_list_max = _get_max_data(
        num_list, acc_list, vars_list, hit_list, all_num_list
    )

    var_list_del = _get_del_var_list(var_list_max)

    all_data = {'num_list': num_list, 'acc_list': acc_list, 'vars_list': vars_list}
    max_acc_data = {
        'num_list_max': num_list_max, 'acc_list_max': acc_list_max, 'var_list_max': var_list_max,
        'hit_list_max': hit_list_max, 'all_num_list_max': all_num_list_max,
    }

    return all_data, max_acc_data, var_list_del


def get_dates(path_in):
    folder_list = os.listdir(path_in)
    date_list = [folder for folder in folder_list if os.path.isdir(path_in + folder) and len(folder) == 16]
    return date_list


def _read_raw_txt_file(file_path):
    lines_list = []
    with open(file_path, 'r') as f_in:
        for line in f_in.readlines():
            lines_list.append(line.replace('\n', ''))
    return lines_list


def _get_all_data(lines_list):
    num, acc, vars_, hit_, all_num_ = [], [], [], [], []

    for line in lines_list[1:]:
        data_tmp = line.split(',')
        num.append(int(data_tmp[0]))
        acc.append(float(data_tmp[1]))
        vars_.append(data_tmp[4:])
        hit_.append(data_tmp[2])
        all_num_.append(data_tmp[3])

    return num, acc, vars_, hit_, all_num_


def _get_max_data(num, acc, vars_, hit_, all_num_):
    var_list, num_list, acc_list, hit_list, all_num_list = [], [], [], [], []

    for num_ in range(min(num), max(num) + 1):
        idx_ = []
        acc_ = []
        for idx_tmp, n_ in enumerate(num):
            this_num = int(all_num_[idx_tmp])
            if n_ == num_ and this_num >= 10:
                idx_.append(idx_tmp)
                acc_.append(acc[idx_tmp])
        if acc_:
            acc_max = max(acc_)

            for idx_tmp, acc_tmp in enumerate(acc_):
                if acc_tmp == acc_max:
                    idx_max = idx_[idx_tmp]
                    break
            else:
                raise ValueError

            var_left = vars_[idx_max]
            hit_max = hit_[idx_max]
            all_num_max = all_num_[idx_max]
        else:
            acc_max, num_, hit_max, all_num_max, var_left = 0, 0, 0, 0, ''
        var_list.append(var_left)
        num_list.append(num_)
        acc_list.append(acc_max)
        hit_list.append(hit_max)
        all_num_list.append(all_num_max)

    return num_list, acc_list, var_list, hit_list, all_num_list


def _get_del_var_list(var_list_max):
    if var_list_max:
        var_add_list = []
        for var_1, var_2 in zip(var_list_max[0:-1], var_list_max[1:]):
            var_1_ = [x.replace('\n', '') for x in var_1]
            var_2_ = [x.replace('\n', '') for x in var_2]

            var_add = [x for x in var_2_ if x not in var_1_]
            # assert len(var_add) == 1, 'del var not unique'
            var_add_list.append(var_add)

        return var_add_list
    else:
        return []
