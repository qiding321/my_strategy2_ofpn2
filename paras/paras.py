# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import datetime

import util.const
import util.util


class Paras:
    time_now = util.util.get_timenow_str()

    def __init__(self):
        # self.reg_name = 'buy_mean_selected'
        # self.reg_name = 'buy_mean_selected_manually'
        # self.reg_name = 'buy_jump_selected'
        # self.reg_name = 'sell_mean_selected'
        self.reg_name = 'sell_mean_selected_manually'
        # self.reg_name = 'sell_jump_selected'
        # self.reg_name = 'buy_mean_selected_rolling'
        # self.reg_name = 'buy_jump_selected_rolling'
        # self.reg_name = 'buy_jump_selected_rolling_cutoffand3month'
        # self.reg_name = 'buy_jump_selected_rolling_cutoffand3month_strict'
        # self.reg_name = 'buy_jump_qd_manually_selected'
        # self.reg_name = 'buy_jump_cy_manually_selected'
        # self.reg_name = 'buy_jump_yy_manually_selected'
        # self.reg_name = 'sell_mean_selected_rolling'
        # self.reg_name = 'sell_jump_selected_rolling'
        # self.reg_name = 'sell_jump_qd_manually_selected'
        # self.reg_name = 'sell_jump_cy_manually_selected'
        # self.reg_name = 'sell_jump_yy_manually_selected'
        # self.reg_name = 'sell_jump_selected_rolling_cutoffand3month'
        # self.reg_name = 'sell_jump_selected_rolling_cutoffand3month_strict'
        # self.reg_name = 'take_log'
        # self.reg_name = 'buy_mean_subsample'
        # self.reg_name = 'rolling_ms_buy_mean'
        # self.reg_name = 'rolling_ms_buy_jump_10bins'
        # self.reg_name = 'rolling_ms_buy_jump_cutoff'
        # self.reg_name = 'rolling_ms_sell_jump_10bins'
        # self.reg_name = 'rolling_ms_sell_jump_cutoff'
        # self.reg_name = 'one_reg_sell_mean_ms'
        # self.reg_name = 'one_reg_buy_mean_corss'
        # self.reg_name = 'one_reg_sell_jump_ms'
        # self.reg_name = 'truncate_selected3_10min'
        # self.reg_name = 'y_jump'

        self.normalize = False
        # self.normalize = True
        self.divided_std = False
        self.add_const = True
        self.method_paras = MethodParas(util.const.FITTING_METHOD.OLS)
        # self.method_paras = MethodParas(util.const.FITTING_METHOD.LOGIT)
        # self.method_paras = MethodParas(util.const.FITTING_METHOD.PROBIT)
        # self.method_paras = MethodParas(util.const.FITTING_METHOD.GARCH)
        # self.method_paras = MethodParas(util.const.FITTING_METHOD.DECTREE)
        # self.x_vars_para = XvarsParaLog()
        # self.x_vars_para = XvarsParaRaw()
        # self.x_vars_para = XvarsParaRawSell()
        # self.x_vars_para = XvarsParaForJump()
        # self.x_vars_para = XvarsParaForJumpSell()
        # self.x_vars_para = XvarsParaTruncate()
        # self.x_vars_para = XvarsParaTruncate2()
        # self.x_vars_para = XvarsParaTruncate3()
        # self.x_vars_para = XvarsParaBuyMeanSelected()
        # self.x_vars_para = XvarsParaBuyMeanSelectedManually()
        # self.x_vars_para = XvarsParaSellMeanSelected()
        self.x_vars_para = XvarsParaSellMeanSelectedManually()
        # self.x_vars_para = XvarsParaBuyJumpSelected()
        # self.x_vars_para = XvarsParaBuyJumpSelected2()
        # self.x_vars_para = XvarsParaBuyJumpSelectedCutoff3Month()
        # self.x_vars_para = XvarsParaBuyJumpQDSelectedManually()
        # self.x_vars_para = XvarsParaBuyJumpCYSelectedManually()
        # self.x_vars_para = XvarsParaBuyJumpYYSelectedManually()
        # self.x_vars_para = XvarsParaBuyJumpSelectedCutoff3MonthStrict()
        # self.x_vars_para = XvarsParaSellJumpSelected()
        # self.x_vars_para = XvarsParaSellJumpQDManuallySelected()
        # self.x_vars_para = XvarsParaSellJumpCYManuallySelected()
        # self.x_vars_para = XvarsParaSellJumpYYManuallySelected()
        # self.x_vars_para = XvarsParaSellJumpSelectedCutoff3Month()
        # self.x_vars_para = XvarsParaSellJumpSelectedCutoff3MonthStrict()
        # self.y_vars = YvarsParaLog()
        # self.y_vars = YvarsParaRaw()
        self.y_vars = YvarsParaRawSell()
        # self.y_vars = YvarsParaJumpSell()
        # self.y_vars = YvarsParaJump()
        self.truncate_paras = TruncateParas()
        self.decision_tree_paras = DecisionTreeParas()
        self.period_paras = PeriodParas()
        # self.period_paras = PeriodParasRolling()
        # self.time_scale_paras = TimeScaleParas('10min', '10min')
        self.time_scale_paras = TimeScaleParas()
        self.high_freq_jump_para = HighFreqParas()

    def __str__(self):
        s = '{reg_name}\n{period}\n{time_scale}\nnormalize: {normalize}\ndivide_std: {divide_std}\nmethod: {method}\n' \
            'x vars: {xvars}\ny vars: {yvars}\n' \
            '{truncate}\n{decision_tree}\n{hfpara}' \
            .format(
            reg_name=self.reg_name,
            period=self.period_paras, time_scale=self.time_scale_paras,
            normalize=self.normalize, divide_std=self.divided_std, method=self.method_paras,
            xvars=self.x_vars_para, yvars=self.y_vars, truncate=self.truncate_paras,
            decision_tree=self.decision_tree_paras,
            hfpara=self.high_freq_jump_para,
        )
        return s

    def get_title(self):
        s = '{reg_name}_{period}_normalize_{normalize}_divide_std_{divide_std}_{method}_{truncate}_{decision_tree}'.format(
            reg_name=self.reg_name, period=self.period_paras, normalize='T' if self.normalize else 'F', divide_std='T' if self.divided_std else 'F', method=self.method_paras,
            truncate=self.truncate_paras, decision_tree=self.decision_tree_paras
        )
        title = self.time_now + s
        return title


class ParasModelSelection(Paras):
    def __init__(self):
        Paras.__init__(self)
        self.reg_name += '_ms'
        self.model_selection = True


class MethodParas:
    def __init__(self, method=None):
        if method is None:
            # self.method = util.const.FITTING_METHOD.ADABOOST
            # self.method = util.const.FITTING_METHOD.LOGIT
            self.method = util.const.FITTING_METHOD.OLS
            # self.method = util.const.FITTING_METHOD.GARCH
            # self.method = util.const.FITTING_METHOD.DECTREE
        else:
            self.method = method

    def __str__(self):
        s = self.method
        return s


class TruncateParas:
    def __init__(self, truncate_bool=True):
        self.truncate = truncate_bool
        self.truncate_method = 'mean_std'
        self.truncate_window = 30
        self.truncate_std = 4

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
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            'asize2_ret_dummy_cross',
            'asize1_ret_dummy_cross',
            'bsize2_ret_dummy_cross',
            'bsize1_ret_dummy_cross',
            'asize1_change_ret_dummy_cross',
            'bsize1_change_ret_dummy_cross',
        ]  # todo
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list + self.cross_term_list
        ))

    def __str__(self):
        s = ', '.join(self.x_vars_list)
        return s


class XvarsParaRawSell:
    def __init__(self):
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'asize1_change',
            'bsize1_change',
            'asize1', 'asize2',
            'bsize1', 'bsize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            'ret_hs300',
            'ret_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            'asize2_ret_dummy_cross',
            'asize1_ret_dummy_cross',
            'bsize2_ret_dummy_cross',
            'bsize1_ret_dummy_cross',
            'asize1_change_ret_dummy_cross',
            'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            'buyvolume_jump_freq_60s',
            'sellvolume_jump_freq_3s',
            'sellvolume_jump_freq_30s',
            'sellvolume_jump_freq_60s',
            'volume_index_sh50_jump_freq_3s',
            'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            'volume_index_hs300_jump_freq_3s',
            'volume_index_hs300_jump_freq_30s',
            'volume_index_hs300_jump_freq_60s',
            'ret_index_index_future_300_jump_freq_3s',
            'ret_index_index_future_300_jump_freq_30s',
            'ret_index_index_future_300_jump_freq_60s',
            'ret_index_index_future_300_abs_jump_freq_3s',
            'ret_index_index_future_300_abs_jump_freq_30s',
            'ret_index_index_future_300_abs_jump_freq_60s',
            'ret_sh50_jump_freq_3s',
            'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            'ret_sh50_abs_jump_freq_3s',
            'ret_sh50_abs_jump_freq_30s',
            'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaForJump(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'bsize1_change',
            'asize2',
            'buyvolume',
            # 'sellvolume',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            'buyvolume_lag4',
            # 'sellvolume_lag2',
        ]
        self.jump_freq_list = [
            'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',

        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list + self.jump_freq_list
        ))


class XvarsParaForJumpSell(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            'ret_sh50',
            'ret_hs300',
            'bsize1_change',
            'asize1_change',
            'asize2', 'asize1',
            'bsize2', 'bsize1',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.moving_average_list = []
        self.high_order_var_list = []
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
            # 'sellvolume_lag5',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.jump_freq_list = [
            'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            'buyvolume_jump_freq_60s',
            'sellvolume_jump_freq_3s',
            'sellvolume_jump_freq_30s',
            'sellvolume_jump_freq_60s',
            'volume_index_sh50_jump_freq_3s',
            'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            'volume_index_hs300_jump_freq_3s',
            'volume_index_hs300_jump_freq_30s',
            'volume_index_hs300_jump_freq_60s',
            'ret_index_index_future_300_jump_freq_3s',
            'ret_index_index_future_300_jump_freq_30s',
            'ret_index_index_future_300_jump_freq_60s',
            'ret_index_index_future_300_abs_jump_freq_3s',
            'ret_index_index_future_300_abs_jump_freq_30s',
            'ret_index_index_future_300_abs_jump_freq_60s',
            'ret_sh50_jump_freq_3s',
            'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            'ret_sh50_abs_jump_freq_3s',
            'ret_sh50_abs_jump_freq_30s',
            'ret_sh50_abs_jump_freq_60s',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list + self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list + self.jump_freq_list
        ))


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


class XvarsParaBuyMeanSelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'asize1_change',
            'bsize1_change',
            # 'asize1',
            'asize2',
            # 'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            'ret_sh50',
            'ret_hs300',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            'asize2_ret_dummy_cross',
            'asize1_ret_dummy_cross',
            'bsize2_ret_dummy_cross',
            'bsize1_ret_dummy_cross',
            'asize1_change_ret_dummy_cross',
            'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
            # 'sellvolume_lag3',
            # 'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyMeanSelectedManually(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            # 'asize1_change',
            # 'asize2_change',
            # 'asize3_change',
            # 'asize4_change',
            # 'asize5_change',
            'bsize1_change',
            # 'bsize2_change',
            # 'bsize3_change',
            # 'bsize4_change',
            # 'asize1',
            'asize2',
            # 'asize3',
            # 'bsize1',
            # 'bsize2',
            # 'bsize3',
            'buyvolume',
            'sellvolume',
            # 'ret_sh50',
            # 'ret_hs300',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days',
            # 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days',
            # 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            # 'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            # 'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellMeanSelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'asize1_change',
            'bsize1_change',
            # 'asize1', 'asize2',
            'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'ret_sh50',
            'ret_hs300',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            'asize1_ret_dummy_cross',
            'bsize2_ret_dummy_cross',
            'bsize1_ret_dummy_cross',
            'asize1_change_ret_dummy_cross',
            'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            'buyvolume_mean5days',
            # 'buyvolume_mean20days',
            'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days',
            'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            'sellvolume_jump_freq_3s',
            'sellvolume_jump_freq_30s',
            'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellMeanSelectedManually(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            'ret_index_index_future_300',
            'asize1_change',
            # 'asize2_change',
            # 'asize3_change',
            # 'asize4_change',
            # 'asize5_change',
            # 'bsize1_change',
            # 'bsize2_change',
            # 'bsize3_change',
            # 'bsize4_change',
            # 'asize1',
            # 'asize2',
            # 'asize3',
            # 'bsize1',
            'bsize2',
            # 'bsize3',
            'buyvolume',
            'sellvolume',
            # 'ret_sh50',
            # 'ret_hs300',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days',
            # 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days',
            # 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            # 'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpSelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            'bsize1_change',
            # 'asize1',
            'asize2',
            'bsize1',
            # 'bsize2',
            'buyvolume',
            'sellvolume',
            # 'ret_hs300',
            'ret_sh50',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpSelected2(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            # 'bsize1_change',
            # 'asize1',
            'asize2',
            'bsize1',
            # 'bsize2',
            'buyvolume',
            # 'sellvolume',
            # 'ret_hs300',
            'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpQDSelectedManually(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            'bsize1_change',
            # 'asize1',
            'asize2',
            'bsize1',
            # 'bsize2',
            'buyvolume',
            # 'sellvolume',
            # 'ret_hs300',
            'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpCYSelectedManually(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            'bsize1_change',
            'asize1',
            # 'asize2',
            'bsize1',
            'bsize2',
            'buyvolume',
            # 'sellvolume',
            # 'ret_hs300',
            'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpYYSelectedManually(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            # 'bsize1_change',
            'asize1',
            'asize2',
            'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            'ret_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpSelectedCutoff3Month(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            'asize1_change',
            # 'bsize1_change',
            'asize1',
            'asize2',
            # 'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            # 'ret_hs300',
            'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaBuyJumpSelectedCutoff3MonthStrict(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            # 'bsize1_change',
            # 'asize1',
            # 'asize2',
            # 'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            # 'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            # 'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellJumpSelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            'asize1_change',
            # 'bsize1_change',
            'asize1',
            # 'asize2',
            # 'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellJumpQDManuallySelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            'asize1_change',
            # 'bsize1_change',
            'asize1',
            # 'asize2',
            # 'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellJumpCYManuallySelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            'asize1_change',
            # 'bsize1_change',
            'asize1',
            # 'asize2',
            'bsize1',
            'bsize2',
            'buyvolume',
            # 'sellvolume',
            'volume_index_sh50',
            'ret_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            # 'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellJumpYYManuallySelected(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            # 'bsize1_change',
            'asize1',
            'asize2',
            'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'volume_index_sh50',
            'ret_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            # 'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellJumpSelectedCutoff3Month(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            'asize1_change',
            'bsize1_change',
            # 'asize1',
            'asize2',
            'bsize1',
            'bsize2',
            'buyvolume',
            'sellvolume',
            'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            'buyvolume_lag3',
            'buyvolume_lag4',
            'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            'sellvolume_jump_freq_30s',
            'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            # 'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
        ))


class XvarsParaSellJumpSelectedCutoff3MonthStrict(XvarsParaRaw):
    def __init__(self):
        XvarsParaRaw.__init__(self)
        self.x_vars_normal_list = [
            # 'ret_index_index_future_300',
            # 'asize1_change',
            # 'bsize1_change',
            # 'asize1',
            'asize2',
            'bsize1',
            'bsize2',
            # 'buyvolume',
            # 'sellvolume',
            'ret_sh50',
            # 'volume_index_sh50',
            # 'volatility_index300_60s',
        ]
        self.cross_term_list = [  # ret_dummy = 1 if ret changes else 0
            # 'asize2_ret_dummy_cross',
            # 'asize1_ret_dummy_cross',
            # 'bsize2_ret_dummy_cross',
            # 'bsize1_ret_dummy_cross',
            # 'asize1_change_ret_dummy_cross',
            # 'bsize1_change_ret_dummy_cross',
        ]  # todo

        self.moving_average_list = [
            # 'buy_vol_10min_intraday_pattern_20_days', 'sell_vol_10min_intraday_pattern_20_days',
            # 'buyvolume_mean5days', 'buyvolume_mean20days',
            # 'buyvolume_mean1day',
            # 'sellvolume_mean5days',
            # 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days',
            # 'volume_index_hs300_mean20days',
            # 'volume_index_hs300_mean1day',
        ]
        self.high_order_var_list = [
            # 'buyvolume', 'sellvolume', 'bsize1_change', 'asize1_change',
            # 'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
            # 'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
            # 'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
            # 'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
        ]
        self.intraday_pattern_list = []
        self.truncate_list = []
        self.log_change_list = []
        self.lag_list = [
            # 'buyvolume_lag2',
            # 'buyvolume_lag3',
            # 'buyvolume_lag4',
            'sellvolume_lag2',
            # 'sellvolume_lag3',
            'sellvolume_lag4',
        ]
        self.log_list = [
        ]
        self.jump_list = []

        self.jump_freq_list = [
            # 'buyvolume_jump_freq_3s',
            # 'buyvolume_jump_freq_30s',
            # 'buyvolume_jump_freq_60s',
            # 'sellvolume_jump_freq_3s',
            # 'sellvolume_jump_freq_30s',
            # 'sellvolume_jump_freq_60s',
            # 'volume_index_sh50_jump_freq_3s',
            # 'volume_index_sh50_jump_freq_30s',
            # 'volume_index_sh50_jump_freq_60s',
            # 'volume_index_hs300_jump_freq_3s',
            # 'volume_index_hs300_jump_freq_30s',
            # 'volume_index_hs300_jump_freq_60s',
            # 'ret_index_index_future_300_jump_freq_3s',
            # 'ret_index_index_future_300_jump_freq_30s',
            # 'ret_index_index_future_300_jump_freq_60s',
            # 'ret_index_index_future_300_abs_jump_freq_3s',
            # 'ret_index_index_future_300_abs_jump_freq_30s',
            'ret_index_index_future_300_abs_jump_freq_60s',
            # 'ret_sh50_jump_freq_3s',
            # 'ret_sh50_jump_freq_30s',
            # 'ret_sh50_jump_freq_60s',
            # 'ret_sh50_abs_jump_freq_3s',
            # 'ret_sh50_abs_jump_freq_30s',
            'ret_sh50_abs_jump_freq_60s',
        ]

        self.x_vars_list = list(set(
            self.x_vars_normal_list + self.moving_average_list +
            self.high_order_var_list + self.intraday_pattern_list +
            self.truncate_list + self.lag_list + self.log_list +
            self.jump_freq_list + self.cross_term_list
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


class YvarsParaRawSell:
    def __init__(self):
        self.y_vars_list = ['sellvolume']
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


class YvarsParaJumpSell(YvarsParaRaw):
    def __init__(self):
        YvarsParaRaw.__init__(self)
        self.y_vars_list = ['sellvolume_jump']


class PeriodParas:
    def __init__(self, begin_training=None, end_training=None, begin_testing=None, end_testing=None):

        self.begin_date = '20130801'
        # self.begin_date = '20140731'
        self.end_date = '20160731'
        # self.end_date = '20150731'

        if begin_training is None:
            # self.begin_date_training = '20130801'
            self.begin_date_training = '20150731'
            # self.end_date_training = '20150731'
            self.end_date_training = '20160731'
            # self.end_date_training = '20140731'
            # self.begin_date_predict = '20150801'
            self.begin_date_predict = '20160801'
            self.end_date_predict = '20160901'
            # self.begin_date_predict = '20140801'
            # self.end_date_predict = '20150731'
        else:
            self.begin_date_training = begin_training
            self.end_date_training = end_training
            self.begin_date_predict = begin_testing
            self.end_date_predict = end_testing

        self.training_period = '12M'
        # self.testing_period = '1M'
        self.testing_period = '3M'
        self.rolling_period = '1M'
        self.testing_demean_period = '12M'

        self.fixed = False

        if self.begin_date_training == '' or self.begin_date_training is None:
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
            s = ''  # todo
        if self.fixed:
            s += '_fixed'
        return s


class ParasModelSelectionRolling(Paras):
    def __init__(self):
        Paras.__init__(self)
        self.begin_date = '20130801'
        self.end_date = '20160731'
        self.training_period = '12M'
        # self.testing_period = '1M'
        self.testing_period = '3M'
        self.rolling_period = '1M'

    def rolling_paras(self):
        training_period = self.training_period
        testing_period = self.testing_period
        date_begin = datetime.datetime.strptime(self.begin_date, '%Y%m%d')
        date_end = datetime.datetime.strptime(self.end_date, '%Y%m%d')

        offset_training = util.util.get_offset(training_period)
        offset_predict = util.util.get_offset(testing_period)
        offset_rolling = util.util.get_offset(self.rolling_period)

        offset_one_day = util.util.get_offset('1D')
        date_moving = date_begin

        training_date_begin_list = []
        training_date_end_list = []
        predict_date_begin_list = []
        predict_date_end_list = []

        while True:
            training_date_begin = date_moving
            training_date_end = date_moving + offset_training
            predict_date_begin = training_date_end + offset_one_day
            predict_date_end = predict_date_begin + offset_predict

            training_date_begin_list.append(training_date_begin.strftime('%Y%m%d'))
            training_date_end_list.append(training_date_end.strftime('%Y%m%d'))
            predict_date_begin_list.append(predict_date_begin.strftime('%Y%m%d'))
            predict_date_end_list.append(predict_date_end.strftime('%Y%m%d'))

            if predict_date_end >= date_end:
                break
            date_moving = date_moving + offset_rolling + offset_one_day

        date_list = list(zip(
            training_date_begin_list, training_date_end_list, predict_date_begin_list, predict_date_end_list
        ))
        paras_list = []
        for d1, d2, d3, d4 in date_list:
            para_tmp = ParasModelSelection()
            para_tmp.period_paras = PeriodParas(d1, d2, d3, d4)
            paras_list.append(para_tmp)

        return paras_list

    def get_title(self):
        s = '{reg_name}_{period}_normalize_{normalize}_divide_std_{divide_std}_{method}_{truncate}_{decision_tree}'.format(
            reg_name=self.reg_name, period=self.begin_date + self.end_date,
            normalize='T' if self.normalize else 'F',
            divide_std='T' if self.divided_std else 'F', method=self.method_paras,
            truncate=self.truncate_paras, decision_tree=self.decision_tree_paras
        )
        title = self.time_now + s
        return title


class PeriodParasRolling(PeriodParas):
    def __init__(self):

        PeriodParas.__init__(self)

        self.begin_date = '20130801'
        self.end_date = '201608301'

        self.begin_date_training = ''
        self.end_date_training = ''
        self.begin_date_predict = ''
        self.end_date_predict = ''

        self.training_period = '12M'
        self.testing_period = '1M'
        self.testing_demean_period = '12M'

        if self.begin_date_training == '' or self.begin_date_training is None:
            self.mode = 'rolling'
        else:
            self.mode = 'one_reg'

    def __str__(self):
        if self.mode == 'rolling':
            # s = 'training{}_predicting{}'.format(self.training_period, self.testing_period)
            s = 'roll_{}_{}'.format(self.training_period, self.testing_period)
        else:
            s = 'one_reg_{}_{}_{}_{}'.format(
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


class HighFreqParas:
    def __init__(self):
        self.freq = '3s'
        self.window = 60
        self.std = 4

    def __str__(self):
        s = 'hfparas_{}by{}for{}std'.format(self.freq, self.window, self.std)
        return s
