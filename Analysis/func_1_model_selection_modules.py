# -*- coding: utf-8 -*-
"""
Created on 2016/11/22 10:29

@author: qiding
"""

import re

import pandas as pd

import util.const


class OneRegResult:
    def __init__(self, reg_type=None, var_list=None, result_path=None, coef=None, z_value=None, p_value=None, in_sample_percentile=None, out_of_sample_percentile=None, para_type=None):
        self.reg_type = '' if reg_type is None else reg_type
        self.var_list = [] if var_list is None else var_list
        self.result_path = '' if result_path is None else result_path
        self.coef = dict() if coef is None else coef
        self.z_value = dict() if z_value is None else z_value
        self.p_value = dict() if p_value is None else p_value
        self.in_sample_percentile = pd.DataFrame() if in_sample_percentile is None else in_sample_percentile
        self.out_of_sample_percentile = pd.DataFrame() if out_of_sample_percentile is None else out_of_sample_percentile

        self.accuracy = 0.0
        self.target_num_oos_largest_group = 0
        self.all_num_oos_largest_group = 0
        self.daily_rsquared = pd.DataFrame()
        self.r_squared = 0.0
        self.para_type = para_type

    def __str__(self):
        s = 'reg type: {}\nvars: {}\np_value: {}\nout-of-sample percentile:\n{}'.format(self.reg_type, self.var_list, self.p_value, self.out_of_sample_percentile.iloc[-1, :])
        return s

    def __repr__(self):
        return self.__str__()

    def update_vars_from_path(self):
        assert self.result_path is not ''
        assert self.reg_type is not ''
        if self.reg_type == util.const.FITTING_METHOD.LOGIT or self.reg_type == util.const.FITTING_METHOD.PROBIT:
            try:
                self._update_var_list()
                self._update_reg_paras('coef')
                self._update_reg_paras('z_value')
                self._update_reg_paras('p_value')
                self._update_percentile('in_sample')
                self._update_percentile('out_of_sample')
                self._update_accuracy()
            except FileNotFoundError:
                raise FileNotFoundError
        elif self.reg_type == util.const.FITTING_METHOD.OLS:
            try:
                self._update_var_list()
                self._update_reg_paras('coef')
                self._update_reg_paras('z_value')
                self._update_reg_paras('p_value')
                self._update_r_squared()
            except FileNotFoundError:
                raise FileNotFoundError

        else:
            raise ValueError

    def is_efficient(self):
        if self.reg_type == util.const.FITTING_METHOD.OLS:
            return True
        elif self.reg_type == util.const.FITTING_METHOD.LOGIT or self.reg_type == util.const.FITTING_METHOD.PROBIT:
            return self.all_num_oos_largest_group >= 10

    def _update_var_list(self):
        var_record_path = self.result_path + 'vars_record.txt'
        var_list = []
        import os
        if os.path.exists(var_record_path):
            with open(var_record_path, 'r') as f_in:
                for line in f_in.readlines():
                    this_var_ = line.replace('\n', '').replace('\t', '')
                    if this_var_ is not '':
                        var_list.append(this_var_)
        else:
            var_record_path = self.result_path + 'variance_training.csv'
            var_list = [var_+'_x' for var_ in list(pd.read_csv(var_record_path, index_col=[0], header=None).index)]
        self.var_list = var_list

    def _update_reg_paras(self, reg_para_name):

        num = {'coef': 1, 'z_value': 3, 'p_value': 4}[reg_para_name]
        number_pattern = '\\s+((-?\\d+(\\.\\d+)?(e-?\\+?\\d+)?)|(nan))'

        reg_para_path = self.result_path + 'reg_summary.txt'
        with open(reg_para_path, 'r') as f_in:
            flag = False if self.para_type == 'marginal_effect' else True
            lines = []
            for one_line in f_in.readlines():
                if not flag:
                    if one_line.find('Marginal Effects') >= 0:
                        flag = True
                if flag:
                    lines.append(one_line)

            content = ''.join(lines)

        reg_para = dict()
        for var_ in self.var_list:
            try:
                number_pattern_multiple = number_pattern * num
                number_ = re.search(pattern='(?<='+var_+')'+number_pattern_multiple, string=content).group()
                numbers_list_ = [tmp_ for tmp_ in number_.split(' ') if tmp_ is not '' and tmp_ is not ' ']
                reg_para[var_] = numbers_list_[num-1]
            except Exception as e:
                print('error: {} ,{} @ {}'.format(str(e), var_, self.result_path))
        if reg_para_name == 'coef':
            self.coef = reg_para
        elif reg_para_name == 'z_value':
            self.z_value = reg_para
        elif reg_para_name == 'p_value':
            self.p_value = reg_para
        else:
            raise ValueError

    def _update_percentile(self, percentile_type):
        try:
            if percentile_type == 'in_sample':
                file_name = 'var_analysis\\in_samplepercentile.csv'
                my_path = self.result_path + file_name
                self.in_sample_percentile = pd.read_csv(my_path)
            elif percentile_type == 'out_of_sample':
                file_name = 'var_analysis\\out_of_samplepercentile.csv'
                my_path = self.result_path + file_name
                self.out_of_sample_percentile = pd.read_csv(my_path)
            else:
                raise ValueError
        except Exception as e:
            if percentile_type == 'in_sample':
                file_name = 'var_analysis\\in_sample10percentile.csv'
                my_path = self.result_path + file_name
                self.in_sample_percentile = pd.read_csv(my_path)
            elif percentile_type == 'out_of_sample':
                file_name = 'var_analysis\\out_of_sample10percentile.csv'
                my_path = self.result_path + file_name
                self.out_of_sample_percentile = pd.read_csv(my_path)
            else:
                raise ValueError


    def _update_accuracy(self):
        self.accuracy = self.out_of_sample_percentile.iloc[-1, :]['accuracy']
        self.target_num_oos_largest_group = self.out_of_sample_percentile.iloc[-1, :]['target_num']
        self.all_num_oos_largest_group = self.out_of_sample_percentile.iloc[-1, :]['all_num']

    def _update_r_squared(self):
        r_squared_path = self.result_path + 'daily_rsquared.csv'
        daily_rsquared = pd.read_csv(r_squared_path, skiprows=[1,2])
        mse = daily_rsquared['mse'].sum()
        msr = daily_rsquared['msr'].sum()
        r_squared = 1 - msr / mse
        self.daily_rsquared = daily_rsquared
        self.r_squared = r_squared
        self.accuracy = r_squared


if __name__ == '__main__':
    this_path = r'E:\StrategyResult\MarketMaking\2016-11-16-15-27-53rolling_ms_2014030120160731_normalize_F_divide_std_F_Logit_truncate_period30_std4_\2014020120150131\7\6\\'
    reg_result = OneRegResult(result_path=this_path, reg_type=util.const.FITTING_METHOD.LOGIT)
    reg_result.update_vars_from_path()


