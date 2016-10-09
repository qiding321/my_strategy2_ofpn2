# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 18:00

@author: qiding
"""

import pandas as pd
import statsmodels.api as sm


class RegMethodWrapper:
    def __init__(self, endog, exog):
        assert isinstance(endog, pd.DataFrame)
        assert isinstance(exog, pd.DataFrame)
        self.endog = endog
        self.exog = exog
        self.paras_reg = None

    def fit(self):
        pass

    def predict(self, exog_new):
        pass


class OLSWrapper(RegMethodWrapper):
    def __init__(self, endog, exog, has_const):
        RegMethodWrapper.__init__(self, endog, exog)
        self.has_const = has_const
        self.model = sm.OLS(endog, sm.add_constant(exog) if self.has_const else exog, hasconst=has_const)

    def fit(self):
        self.paras_reg = self.model.fit()
        return self.paras_reg

    def predict(self, exog_new):
        assert isinstance(exog_new, pd.DataFrame)
        predict_result = self.model.predict(params=self.paras_reg.params, exog=sm.add_constant(exog_new) if self.has_const else exog_new)
        return predict_result


class LogitWrapper(RegMethodWrapper):
    def __init__(self, endog, exog):
        RegMethodWrapper.__init__(self, endog, exog)

    def fit(self):
        pass

    def predict(self, exog_new):
        pass
