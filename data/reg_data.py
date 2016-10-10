# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import log.log
import method_wrapper.reg_method_wrapper
import paras.paras
import util.const
import util.util

my_log = log.log.log_order_flow_predict


class RegData:
    def __init__(self, x_vars, y_vars,
                 x_vars_before_normalize, y_vars_before_normalize,
                 paras_config, normalize_funcs):
        assert isinstance(x_vars, pd.DataFrame)
        assert isinstance(x_vars, pd.DataFrame)
        assert isinstance(x_vars_before_normalize, pd.DataFrame)
        assert isinstance(y_vars_before_normalize, pd.DataFrame)
        assert isinstance(paras_config, paras.paras.Paras)
        assert isinstance(normalize_funcs, dict)

        self.x_vars = x_vars
        self.y_vars = y_vars

        self.x_vars_raw = x_vars_before_normalize
        self.y_vars_raw = y_vars_before_normalize

        self.paras_config = paras_config
        self.normalize_funcs = normalize_funcs

        self.paras_reg = None
        self.model = None
        self.y_predict = None
        self.y_predict_before_normalize = None
        self.reg_data_training = None

    def predict(self):
        y_predict = self.model.predict(exog_new=self.x_vars)
        self.y_predict = y_predict
        normalize_funcs_reverse = self.normalize_funcs['y_series_normalize_func_reverse']
        self.y_predict_before_normalize = normalize_funcs_reverse(self.y_predict)

    def report_err_decomposition(self, output_path, file_name, predict_period):  # todo
        err_dict = self._err_decomposition()
        assert isinstance(err_dict, dict)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            my_log.info('make dirs: {}'.format(output_path))

        new_dict = {}
        for k, v in err_dict.items():
            if k == 'variance_x':
                for k_, v_ in v.items():
                    new_dict[k_ + 'var_x'] = v_
            elif k == 'variance_x_contribution':
                for k_, v_ in v.items():
                    new_dict[k_ + 'var_contrb_x'] = v_
            elif k == 'rsquared_out_of_sample' or k == 'rsquared_out_of_sample_by_oos_mean':
                new_dict[k] = v
            elif k == 'ssr' or k == 'sse' or k == 'sse_by_oos_mean':
                new_dict[k] = v
            else:
                new_dict[k] = v
        new_df = pd.DataFrame(pd.Series(new_dict), columns=[predict_period])

        if os.path.exists(output_path + file_name):
            data_exist = pd.read_csv(output_path + file_name, index_col=[0])
            data_to_rcd = pd.merge(new_df, data_exist, left_index=True, right_index=True)
        else:
            data_to_rcd = new_df

        data_to_rcd.sort_index(axis=1).to_csv(output_path + file_name)

    def report_daily_rsquared(self, output_path, file_name):
        data_merged = self._get_y_predict_merged()

        def _generate_one_day_stats(c):
            mse_ = (c['sse'] * c['sse']).sum()
            msr_ = (c['error'] * c['error']).sum()
            r_sq_ = 1 - msr_ / mse_
            ret_ = pd.DataFrame([mse_, msr_, r_sq_], index=['mse', 'msr', 'rsquared']).T
            return ret_

        r_squared_daily = data_merged.groupby('ymd').apply(_generate_one_day_stats).unstack()
        r_squared_daily.to_csv(output_path + file_name)

    def plot_daily_fitting(self, output_path):
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
        data_merged = self._get_y_predict_merged()
        for key, data_one_day in data_merged.groupby('ymd'):
            fig = plt.figure()
            plt.plot(data_one_day['y_raw'].values, 'r-')
            plt.plot(data_one_day['y_predict'].values, 'b-')
            fig.savefig(output_path + 'predict_volume_vs_raw_volume' + '-'.join([str(k_) for k_ in key]) + '.jpg')
            plt.close()
            fig = plt.figure()

            plt.scatter(data_one_day['y_raw'], data_one_day['y_predict'], color='b')
            minmin = min(data_one_day['y_raw'].min(), data_one_day['y_predict'].min())
            maxmax = max(data_one_day['y_raw'].max(), data_one_day['y_predict'].max())
            plt.plot([minmin, maxmax], [minmin, maxmax], 'r-')
            plt.xlabel('y_raw')
            plt.ylabel('y_predict')
            fig.savefig(output_path + 'scatter' + '-'.join([str(k_) for k_ in key]) + '.jpg')
            plt.close()

    def plot_error_hist(self, output_path, file_name):
        data_merged = self._get_y_predict_merged()
        error_this_month = data_merged['error']
        plt.hist(error_this_month.values, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.savefig(output_path + file_name)
        plt.close()

    def record_error_description(self, output_path, file_name):
        data_merged = self._get_y_predict_merged()
        error_this_month = data_merged['error']

        err_des = error_this_month.describe()
        err_des['skew'] = error_this_month.skew()
        err_des['kurt'] = error_this_month.kurt()

        err_des.to_csv(output_path + file_name)

    def plot_y_var_hist(self, output_path, file_name):
        y_var = self.y_vars_raw.values
        plt.hist(y_var, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.savefig(output_path + file_name)
        plt.close()

    def _get_y_predict_merged(self):
        y_raw = self.y_vars_raw
        y_training = self.reg_data_training.y_vars_raw.values.T[0]  # todo
        y_predict = pd.DataFrame(self.y_predict_before_normalize, index=y_raw.index, columns=['y_predict'])
        data_merged = pd.merge(y_raw, y_predict, left_index=True, right_index=True).rename(
            columns={y_raw.columns[0]: 'y_raw', y_predict.columns[0]: 'y_predict'})
        data_merged['ymd'] = list(map(lambda x: (x.year, x.month, x.day), data_merged.index))
        data_merged['error'] = data_merged['y_raw'] - data_merged['y_predict']
        data_merged['sse'] = data_merged['y_raw'] - y_training.mean()
        return data_merged

    def _err_decomposition(self):
        y_actual = self.y_vars.values.T[0]
        y_predict = self.y_predict
        y_training = self.model.endog.values.T[0]
        assert isinstance(y_actual, np.ndarray)
        assert isinstance(y_predict, np.ndarray)
        assert isinstance(y_training, np.ndarray)
        assert y_actual.shape == y_predict.shape

        ssr = y_predict - y_actual
        sse = y_actual - y_training.mean()  # for y_mean_in_sample, new
        rsquared = 1 - (ssr * ssr).sum() / (sse * sse).sum()
        var_y = y_actual.var()
        var_y_predict = y_predict.var()
        bias_squared = ((y_predict - y_actual.mean()) * (y_predict - y_actual.mean())).mean()
        bias_mean = y_predict.mean() - y_actual.mean()
        cov_y_y_predict_multiplied_by_minus_2 = -2 * np.cov([y_actual, y_predict])[0, 1]  # todo

        err_dict = {
            'ssr'                                  : (ssr * ssr).mean(),
            'sse'                                  : (sse * sse).mean(),
            'variance_x'                           : self.x_vars.var(),
            'variance_x_contribution'              : self.x_vars.var() * (self.paras_reg.params * self.paras_reg.params),
            'rsquared_out_of_sample'               : rsquared,
            'var_y'                                : var_y,
            'var_y_predict'                        : var_y_predict,
            'bias_squared'                         : bias_squared,
            'bias_mean'                            : bias_mean,
            'cov_y_y_predict_multiplied_by_minus_2': cov_y_y_predict_multiplied_by_minus_2,
        }

        return err_dict


class RegDataTraining(RegData):
    def __init__(self, x_vars, y_vars,
                 x_vars_before_normalize, y_vars_before_normalize,
                 paras_config, normalize_funcs):
        RegData.__init__(self, x_vars, y_vars,
                         x_vars_before_normalize, y_vars_before_normalize,
                         paras_config, normalize_funcs)
        self.num_of_x_vars = len(x_vars.columns)
        self.x_var_names = x_vars.columns
        self.reg_data_training = None

    def fit(self):
        add_const = self.paras_config.add_const
        method = self.paras_config.method_paras

        if method.method == util.const.FITTING_METHOD.OLS:
            self.model = method_wrapper.reg_method_wrapper.OLSWrapper(self.y_vars, self.x_vars, has_const=add_const)
            self.paras_reg = self.model.fit()
            self.predict()
            return self.paras_reg.rsquared
        elif method.method == util.const.FITTING_METHOD.DECTREE:
            decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
            self.model = DecisionTreeClassifier(max_depth=decision_tree_depth)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict = y_predict_insample
        elif method.method == util.const.FITTING_METHOD.DECTREEREG:
            decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
            self.model = DecisionTreeRegressor(max_depth=decision_tree_depth)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict = y_predict_insample
        elif method.method == util.const.FITTING_METHOD.ADABOOST:
            decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
            rng = np.random.RandomState(1)
            self.model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=decision_tree_depth), n_estimators=300, random_state=rng)
            self.model.fit(self.x_vars, self.y_vars)
            y_predict_insample = self.model.predict(self.x_vars)
            self.y_predict = y_predict_insample
        elif method.method == util.const.FITTING_METHOD.LOGIT:
            self.model = method_wrapper.reg_method_wrapper.LogitWrapper(endog=self.y_vars, exog=self.x_vars)
            self.paras_reg, separator = self.model.fit()
            y_predict_insample = self.model.predict(exog=self.x_vars, params=self.paras_reg.params, separator=separator)
            self.y_predict = y_predict_insample
        else:
            my_log.error('reg_method not found: {}'.format(method))
            raise ValueError


class RegDataTest(RegData):
    def __init__(self, x_vars, y_vars,
                 x_vars_before_normalize, y_vars_before_normalize,
                 paras_config, normalize_funcs, reg_data_training):
        RegData.__init__(self, x_vars, y_vars,
                         x_vars_before_normalize, y_vars_before_normalize,
                         paras_config, normalize_funcs)
        self.reg_data_training = reg_data_training

    def add_model(self, model=None, paras_reg=None):
        assert isinstance(model, method_wrapper.reg_method_wrapper.RegMethodWrapper)
        self.model = model
        self.paras_reg = paras_reg
