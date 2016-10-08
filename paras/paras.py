# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import util.const
import util.util


class Paras:
    time_now = util.util.get_timenow_str()

    def __init__(self):
        self.reg_name = 'log'
        self.normalize = True
        self.divided_std = False
        self.add_const = True
        self.method_paras = MethodParas()
        self.x_vars_para = XvarsPara()
        self.y_vars = YvarsPara()
        self.truncate_paras = TruncateParas()
        self.decision_tree_paras = DecisionTreeParas()
        self.period_paras = PeriodParas()
        self.time_scale_paras = TimeScaleParas()

    def __str__(self):
        s = '{reg_name}\n{period}\n{time_scale}\nnormalize: {normalize}\ndivide_std: {divide_std}\nmethod: {method}\nx vars: {xvars}\ny vars: {yvars}\n' \
            '{truncate}\n{decision_tree}'.format(reg_name=self.reg_name,
                                                 period=self.period_paras, time_scale=self.time_scale_paras,
                                                 normalize=self.normalize, divide_std=self.divided_std, method=self.method_paras,
                                                 xvars=self.x_vars_para, yvars=self.y_vars, truncate=self.truncate_paras, decision_tree=self.decision_tree_paras
                                                 )
        return s

    def get_title(self):
        s = '{reg_name}_{period}_normalize_{normalize}_divide_std_{divide_std}_{method}_{truncate}_{decision_tree}'.format(
            reg_name=self.reg_name, period=self.period_paras, normalize=self.normalize, divide_std=self.divided_std, method=self.method_paras,
            truncate=self.truncate_paras, decision_tree=self.decision_tree_paras
        )
        title = self.time_now + s
        return title


class MethodParas:
    def __init__(self):
        # self.method = util.const.FITTING_METHOD.ADABOOST
        # self.method = util.const.FITTING_METHOD.LOGIT
        self.method = util.const.FITTING_METHOD.OLS
        # self.method = util.const.FITTING_METHOD.DECTREE

    def __str__(self):
        s = self.method
        return s


class TruncateParas:
    def __init__(self):
        self.truncate = True
        self.truncate_method = 'mean_std'
        self.truncate_window = 30
        self.truncate_std = 2

    def __str__(self):
        if self.truncate:
            s = 'truncate_period{}_std{}'.format(self.truncate_window, self.truncate_std)
        else:
            s = ''
        return s


class DecisionTreeParas:
    def __init__(self):
        self.decision_tree = False
        self.decision_tree_depth = 5

    def __str__(self):
        if self.decision_tree:
            s = 'decision_tree_depth{}'.format(self.decision_tree_depth)
        else:
            s = ''
        return s


class XvarsPara:
    def __init__(self):
        self.x_vars_normal_list = [
            'asize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            'ret_index_index_future_300',
            'bsize1_change',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.lag_list = []
        self.jump_list = []

        self.x_vars_list = self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list + self.truncate_list + \
                           self.lag_list

    def __str__(self):
        s = 'x_vars_' + '_'.join(self.x_vars_list)
        return s


class YvarsPara:
    def __init__(self):
        self.y_vars_list = []
        # self.jump_list = []
        # self.truncate_list = []

        # self.y_vars_list = self.y_vars_list_normal + self.jump_list + self.truncate_list

    def __str__(self):
        s = 'y_vars_' + '_'.join(self.y_vars_list)
        return s


class PeriodParas:
    def __init__(self):

        self.begin_date = ''
        self.end_date = ''

        self.begin_date_training = ''
        self.end_date_training = ''
        self.begin_date_predict = ''
        self.end_date_predict = ''

        self.training_period = '12M'
        self.testing_period = '1M'
        self.testing_demean_period = '12M'

        if self.begin_date_training == '':
            self.mode = 'rolling'
        else:
            self.mode = 'one_reg'

    def __str__(self):
        if self.mode == 'rolling':
            s = 'training{}_predicting{}'.format(self.training_period, self.testing_period)
        else:
            s = 'training_begin{}_end{}_predicting_begin{}_end{}'.format(
                self.begin_date_training, self.end_date_training, self.begin_date_predict, self.end_date_predict
            )
        return s


class TimeScaleParas:
    def __init__(self):
        self.time_freq = '3s'
        self.time_scale_x = '1min'
        self.time_scale_y = '1min'

    def __str__(self):
        s = '_'.join([self.time_scale_x, self.time_scale_y])
        return s
