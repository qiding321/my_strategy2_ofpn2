# -*- coding: utf-8 -*-
"""
Created on 2016/11/1 17:06

@author: qiding
"""

import os

import data.data
import data.reg_data
import log.log

my_log = log.log.log_order_flow_predict


class ModelSelection:
    def __init__(self, reg_data_training, reg_data_testing):
        assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
        assert isinstance(reg_data_testing, data.reg_data.RegDataTest)
        self.reg_data_training = reg_data_training
        self.reg_data_testing = reg_data_testing
        self.paras = reg_data_training.paras_config
        self.x_var_names_all = reg_data_training.x_vars.columns
        self.var_left = self.x_var_names_all
        self.model_len = len(self.var_left)
        self.model_num_now = 0
        self.var_to_test = []
        self._first_flag = True

    def iter_model_config(self):
        self.model_num_now = 0
        for var_ in self.var_left:
            var_to_test = [v_ for v_ in self.var_left if v_ != var_]  # todo
            # var_to_test = [var_]  # todo
            self.var_to_test = var_to_test
            x_series_rename_training = self.reg_data_training.x_vars[var_to_test]
            y_series_rename_training = self.reg_data_training.y_vars
            normalize_funcs = self.reg_data_training.normalize_funcs
            reg_data_training = data.reg_data.RegDataTraining(
                x_vars=x_series_rename_training, y_vars=y_series_rename_training,
                x_vars_before_normalize=x_series_rename_training, y_vars_before_normalize=y_series_rename_training,
                paras_config=self.paras, normalize_funcs=normalize_funcs,
            )
            x_series_rename_testing = self.reg_data_testing.x_vars[var_to_test]
            y_series_testing = self.reg_data_testing.y_vars
            reg_data_testing = data.reg_data.RegDataTest(
                x_vars=x_series_rename_testing, y_vars=y_series_testing,
                x_vars_before_normalize=x_series_rename_testing, y_vars_before_normalize=y_series_testing,
                paras_config=self.paras, normalize_funcs=normalize_funcs, reg_data_training=reg_data_training
            )

            self.model_num_now += 1

            yield reg_data_training, reg_data_testing, var_

    def iter_model_len(self):
        # return [1]  # todo
        return range(len(self.x_var_names_all), 0 - 1, -1)

    def del_var(self, var_to_del):
        self.var_left = [v_ for v_ in self.var_left if v_ != var_to_del]
        self.model_len = len(self.var_left)

    def get_name(self, path_):
        return path_ + str(self.model_len) + '\\' + str(self.model_num_now) + '\\'

    def record_vars(self, path_):
        if not os.path.exists(path_):
            os.makedirs(path_)
        with open(path_ + 'vars_record.txt', 'w') as f_out:
            s = '\n'.join(self.var_to_test)
            f_out.write(s)

    def record_result(self, output_path, max_bar_accuracy_oos, max_bar_hit, max_bar_len):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if self._first_flag:
            s = 'model_len,max_bar_accuracy_oos,max_bar_hit,max_bar_len,var_names'
        else:
            s = ''
        self._first_flag = False
        s += '\n{},{},{},{},{}'.format(self.model_len,
                                       max_bar_accuracy_oos,
                                       max_bar_hit,
                                       max_bar_len,
                                       ','.join(self.var_to_test))
        with open(output_path + 'accuracy_record.csv', 'a') as f_out:
            my_log.info(s)
            f_out.write(s)
