# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 18:00

@author: qiding
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier

import log.log

# import statsmodels.sandbox.tsa.garch as garch
my_log = log.log.log_order_flow_predict


class RegMethodWrapper:
    def __init__(self, endog, exog):
        assert isinstance(endog, pd.DataFrame)
        assert isinstance(exog, pd.DataFrame)
        self.endog = endog
        self.exog = exog
        self.paras_reg = None

    def fit(self):
        pass

    def predict(self, exog_new, endg_new=None):
        pass

    def summary(self):
        summary = ''
        try:
            summary += self.paras_reg.summary().__str__()
        except Exception as e:
            my_log.error('no summary record, {}'.format(e))
        try:
            summary += '\n\n\n' + self.paras_reg.get_margeff().summary().__str__()
        except Exception as e:
            my_log.error('no marginal effects record, {}'.format(e))
        return summary


class OLSWrapper(RegMethodWrapper):
    def __init__(self, endog, exog, has_const):
        RegMethodWrapper.__init__(self, endog, exog)
        self.has_const = has_const
        self.model = sm.OLS(
            endog, sm.add_constant(exog, has_constant='raise') if self.has_const else exog,
            hasconst=has_const
        )

    def fit(self):
        self.paras_reg = self.model.fit()
        return self.paras_reg

    def predict(self, exog_new, endg_new=None):
        assert isinstance(exog_new, pd.DataFrame)
        predict_result = self.model.predict(
            params=self.paras_reg.params,
            exog=sm.add_constant(exog_new, has_constant='add') if self.has_const else exog_new)
        return predict_result


class LogitWrapper(RegMethodWrapper):
    def __init__(self, endog, exog, has_const):
        RegMethodWrapper.__init__(self, endog, exog)
        self.has_const = has_const
        self.model = sm.Logit(
            endog=endog,
            exog=sm.add_constant(exog, has_constant='raise') if self.has_const else exog, hasconst=has_const
        )

    def fit(self):
        try:
            self.paras_reg = self.model.fit()  # todo
        except Exception as e:
            my_log.error('newton method generates error, use Nelder Mead method instead')
            self.paras_reg = self.model.fit(method='nm')
        return self.paras_reg

    def predict(self, exog_new, endg_new=None):
        predict_result = self.paras_reg.predict(
            exog=sm.add_constant(exog_new, has_constant='add') if self.has_const else exog_new)
        return predict_result


class ProbitWrapper(RegMethodWrapper):
    def __init__(self, endog, exog, has_const):
        RegMethodWrapper.__init__(self, endog, exog)
        self.has_const = has_const
        self.model = sm.Probit(endog, sm.add_constant(exog, has_constant='raise') if self.has_const else exog,
                               hasconst=has_const)

    def fit(self):
        self.paras_reg = self.model.fit()
        return self.paras_reg

    def predict(self, exog_new, endg_new=None):
        predict_result = self.paras_reg.predict(
            exog=sm.add_constant(exog_new, has_constant='add') if self.has_const else exog_new)
        return predict_result


class DecisionTreeWrapper(RegMethodWrapper):
    def __init__(self, endog, exog, para):
        RegMethodWrapper.__init__(self, endog, exog)
        self.para = para
        self.model = DecisionTreeClassifier(max_depth=para.decision_tree_depth)

    def fit(self):
        self.paras_reg = self.model.fit(self.exog, self.endog)
        return self.paras_reg

    def predict(self, exog_new, endg_new=None):
        predict_result = self.model.predict(exog_new)
        return predict_result


class GarchWrapper(RegMethodWrapper):

    def __init__(self, endog, exog):
        import matlab.engine
        RegMethodWrapper.__init__(self, endog, exog)
        self.mat_eng = matlab.engine.start_matlab()
        my_log.info('Matlab Engine Init')

    def fit(self):
        import matlab
        mat_eng = self.mat_eng

        y_mat = matlab.double(self._df_to_list(self.endog))
        mat_eng.workspace['y'] = y_mat

        x_mat = matlab.double(self._df_to_list(self.exog))
        mat_eng.workspace['x'] = x_mat

        mat_eng.eval('mdl=arima(0,0,0);mdl.Variance=garch(1,1);estmdl=estimate(mdl,y\',\'X\',x\');', nargout=0)

    def predict(self, exog_new, endg_new=None):
        import matlab
        mat_eng = self.mat_eng
        x_mat = matlab.double(self._df_to_list(exog_new))
        my_log.info('exog_names: {}'.format(exog_new.columns))
        mat_eng.workspace['x_oos'] = x_mat

        y_oos_mat = matlab.double(self._df_to_list(endg_new))
        mat_eng.workspace['y_oos'] = y_oos_mat

        predict_str = '[E, V, logL] = infer(estmdl, y_oos\', \'X\', x_oos\');y_oos_predict=y_oos\' - E;'
        mat_eng.eval(predict_str, nargout=0)

        y_predict_mat = mat_eng.workspace['y_oos_predict']
        y_predict = np.array([y_[0] for y_ in y_predict_mat])
        return y_predict

    def predict_var(self):
        var_predict_mat = self.mat_eng.workspace['V']
        var_predict = np.array([y_[0] for y_ in var_predict_mat])
        return var_predict

    def _df_to_list(self, df):
        if isinstance(df, pd.DataFrame):
            data_list = [self._df_to_list(df[col_name]) for col_name in df]
        else:
            assert isinstance(df, pd.Series)
            data_list = [float(v_) for v_ in df.tolist()]
        return data_list

if __name__ == '__main__':
    pass
