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
        self.reg_name = 'take_log'
        # self.reg_name = 'one_reg_raw'
        # self.reg_name = 'truncate_selected3_10min'
        # self.reg_name = 'y_jump'
        self.normalize = False
        # self.normalize = True
        self.divided_std = False
        self.add_const = True
        self.method_paras = MethodParas()
        self.x_vars_para = XvarsParaLog()
        # self.x_vars_para = XvarsParaRaw()
        # self.x_vars_para = XvarsParaTruncate()
        # self.x_vars_para = XvarsParaTruncate2()
        # self.x_vars_para = XvarsParaTruncate3()
        # self.y_vars = YvarsParaLog()
        self.y_vars = YvarsParaRaw()
        # self.y_vars = YvarsParaJump()
        self.truncate_paras = TruncateParas(truncate_bool=False)
        self.decision_tree_paras = DecisionTreeParas()
        self.period_paras = PeriodParas()
        # self.time_scale_paras = TimeScaleParas('10min', '10min')
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
            reg_name=self.reg_name, period=self.period_paras, normalize='T' if self.normalize else 'F', divide_std='T' if self.divided_std else 'F', method=self.method_paras,
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
    def __init__(self, truncate_bool=True):
        self.truncate = truncate_bool
        self.truncate_method = 'mean_std'
        self.truncate_window = 30
        self.truncate_std = 6

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


class XvarsParaRaw:
    def __init__(self):
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'bsize1_change',
            'asize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list
        ))

    def __str__(self):
        s = ', '.join(self.x_vars_list)
        return s


class XvarsParaLog(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'asize2',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.lag_list = [
            'buyvolume_log_lag2',
            'buyvolume_log_lag3',
            'buyvolume_log_lag4',
            'sellvolume_log_lag2',
        ]
        self.log_list = [
            'buyvolume_log',
            'sellvolume_log',
            'volume_index_sh50_log',
            'buyvolume_log_lag2',
            'buyvolume_log_lag3',
            'buyvolume_log_lag4',
            'sellvolume_log_lag2',
        ]
        self.log_change_list = [
            'bsize1_log_change',
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list + self.log_change_list
        ))


class XvarsParaTruncate(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'bsize1_change',
            'asize2',
            'volume_index_sh50',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = [
            'buyvolume_truncate',
            'sellvolume_truncate',
            'buyvolume_lag2_truncate',
            'buyvolume_lag3_truncate',
            'buyvolume_lag4_truncate',
            'sellvolume_lag2_truncate',

        ]
        self.lag_list = [
            'buyvolume_lag2_truncate',
            'buyvolume_lag3_truncate',
            'buyvolume_lag4_truncate',
            'sellvolume_lag2_truncate',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list
        ))


class XvarsParaTruncate2(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'bsize1_change',
            'asize2',
            'volume_index_sh50',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = [
            'buyvolume_truncate',
            'sellvolume_truncate',
            'buyvolume_lag2_truncate',
            'sellvolume_lag2_truncate',

        ]
        self.lag_list = [
            'buyvolume_lag2_truncate',
            'sellvolume_lag2_truncate',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list
        ))


class XvarsParaTruncate3(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'bsize1_change',
            'asize2',
            'volume_index_sh50',
        ]
        self.moving_average_list = [
            'buyvolume_mean1day',
            'buyvolume_mean20days',
        ]
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = [
            'buyvolume_truncate',
            'sellvolume_truncate',
            'buyvolume_lag2_truncate',
            'buyvolume_lag3_truncate',

        ]
        self.lag_list = [
            'buyvolume_lag2_truncate',
            'buyvolume_lag3_truncate',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list
        ))


class YvarsParaRaw:
    def __init__(self):
        self.y_vars_list = ['buyvolume']
        # self.jump_list = []
        # self.truncate_list = []

        # self.y_vars_list = self.y_vars_list_normal + self.jump_list + self.truncate_list

    def __str__(self):
        s = ', '.join(self.y_vars_list)
        return s


class YvarsParaLog(YvarsParaRaw):
    def __init__(self):
        YvarsParaRaw.__init__(self)
        self.y_vars_list = ['buyvolume_log']
        # self.jump_list = []
        # self.truncate_list = []

        # self.y_vars_list = self.y_vars_list_normal + self.jump_list + self.truncate_list


class YvarsParaJump(YvarsParaRaw):
    def __init__(self):
        YvarsParaRaw.__init__(self)
        self.y_vars_list = ['buyvolume_jump']
        # self.jump_list = []
        # self.truncate_list = []

        # self.y_vars_list = self.y_vars_list_normal + self.jump_list + self.truncate_list


class PeriodParas:
    def __init__(self):

        self.begin_date = '20130801'
        self.end_date = '201608301'

        self.begin_date_training = '20130801'
        self.end_date_training = '20150731'
        self.begin_date_predict = '20150801'
        self.end_date_predict = '20160801'

        self.training_period = '12M'
        self.testing_period = '1M'
        self.testing_demean_period = '12M'

        if self.begin_date_training == '':
            self.mode = 'rolling'
        else:
            self.mode = 'one_reg'

    def __str__(self):
        if self.mode == 'rolling':
            # s = 'training{}_predicting{}'.format(self.training_period, self.testing_period)
            s = '{}_{}'.format(self.training_period, self.testing_period)
        else:
            s = '{}_{}_{}_{}'.format(
                self.begin_date_training, self.end_date_training, self.begin_date_predict, self.end_date_predict
            )
        return s


class TimeScaleParas:
    def __init__(self, time_scale_x='1min', time_scale_y='1min'):
        self.time_freq = '3s'
        self.time_scale_x = time_scale_x
        self.time_scale_y = time_scale_y

    def __str__(self):
        s = '_'.join([self.time_scale_x, self.time_scale_y])
        return s
