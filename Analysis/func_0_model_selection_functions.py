# -*- coding: utf-8 -*-
"""
Created on 2016/11/17 11:59

@author: qiding
"""


def read_accuracy_file(file_path):
    lines_list = _read_raw_txt_file(file_path)

    num_list, acc_list, vars_list = _get_all_data(lines_list)

    num_list_max, acc_list_max, var_list_max = _get_max_data(num_list, acc_list, vars_list)

    var_list_del = _get_del_var_list(var_list_max)

    all_data = {'num_list': num_list, 'acc_list': acc_list, 'vars_list': vars_list}
    max_acc_data = {'num_list_max': num_list_max, 'acc_list_max': acc_list_max, 'var_list_max': var_list_max}

    return all_data, max_acc_data, var_list_del


def get_dates(path_in):
    pass  # todo


def _read_raw_txt_file(file_path):
    lines_list = []
    with open(file_path, 'r') as f_in:
        for line in f_in.readlines():
            lines_list.append(line.replace('\n', ''))
    return lines_list


def _get_all_data(lines_list):
    num, acc, vars_ = [], [], []

    for line in lines_list[1:]:
        data_tmp = line.split(',')
        num.append(int(data_tmp[0]))
        acc.append(float(data_tmp[1]))
        vars_.append(data_tmp[4:])

    return num, acc, vars_


def _get_max_data(num, acc, vars_):
    var_list, num_list, acc_list = [], [], []

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

        num_list.append(num_)
        acc_list.append(acc_max)

    return num_list, acc_list, var_list


def _get_del_var_list(var_list_max):
    var_add_list = []
    for var_1, var_2 in zip(var_list_max[0:-1], var_list_max[1:]):
        var_1_ = [x.replace('\n', '') for x in var_1]
        var_2_ = [x.replace('\n', '') for x in var_2]

        var_add = [x for x in var_2_ if x not in var_1_]
        var_add_list.append(var_add)

    return var_add_list
