# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import log.log
import method_wrapper.reg_method_wrapper
import paras.paras
import util.const
import util.util

my_log = log.log.log_order_flow_predict


class RegData:
    def __init__(self, x_vars, y_vars, paras_config):
        assert isinstance(x_vars, pd.DataFrame)
        assert isinstance(x_vars, pd.DataFrame)
        assert isinstance(paras_config, paras.paras.Paras)

        self.x_vars = x_vars
        self.y_vars = y_vars
        self.paras_config = paras_config

        self.paras_reg = None
        self.model = None


class RegDataTraining(RegData):
    def __init__(self, x_vars, y_vars, paras_config):
        RegData.__init__(self, x_vars, y_vars, paras_config)
        self.num_of_x_vars = len(x_vars.columns)
        self.x_var_names = x_vars.columns
        self.y_predict_in_sample = None

    def fit(self):
        add_const = self.paras_config.add_const
        method = self.paras_config.method_paras

        if method.method == util.const.FITTING_METHOD.OLS:
            if add_const:
                x_new = sm.add_constant(self.x_vars)
            else:
                x_new = self.x_vars
            self.model = method_wrapper.reg_method_wrapper.OLSWrapper(self.y_vars, x_new, has_const=add_const)
            # self.model = sm.OLS(self.y_vars, x_new, hasconst=add_const)
            self.paras_reg = self.model.fit()
            self.y_predict_insample = self.model.predict(exog_new=x_new)
            return self.paras_reg.rsquared
        elif method.method == util.const.FITTING_METHOD.DECTREE:
            decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
            self.model = DecisionTreeClassifier(max_depth=decision_tree_depth)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict_insample = y_predict_insample
        elif method.method == util.const.FITTING_METHOD.DECTREEREG:
            decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
            self.model = DecisionTreeRegressor(max_depth=decision_tree_depth)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict_insample = y_predict_insample
        elif method.method == util.const.FITTING_METHOD.ADABOOST:
            decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
            rng = np.random.RandomState(1)
            self.model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=decision_tree_depth), n_estimators=300, random_state=rng)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict_insample = y_predict_insample
        elif method.method == util.const.FITTING_METHOD.LOGIT:
            self.model = method_wrapper.reg_method_wrapper.LogitWrapper(endog=self.y_vars, exog=self.x_vars)
            self.paras_reg, separator = self.model.fit()
            y_predict_insample = self.model.predict(exog=self.x_vars, params=self.paras_reg.params, separator=separator)
            self.y_predict_insample = y_predict_insample
        else:
            my_log.error('reg_method not found: {}'.format(method))
            raise ValueError


class RegDataTest(RegData):
    def __init__(self, x_vars, y_vars, paras_config):
        RegData.__init__(self, x_vars, y_vars, paras_config)
        self.y_predict_out_of_sample = None

    def add_model(self, model=None, paras_reg=None):
        self.model = model
        self.paras_reg = paras_reg

    def predict(self):
        pass
