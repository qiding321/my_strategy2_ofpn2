# -*- coding: utf-8 -*-
"""
Created on 2016/10/13 17:06

@author: qiding
"""

import numpy as np
import statsmodels.api as sm


def main():
    data_len = 100000
    std_err = 0.1

    x_var = gen_x_var(data_len, std=1)
    error = gen_error(data_len, std=std_err)
    y_var = func(x_var, error)

    model = ModelLogNormal(y_var, x_var)
    model.fit()

    data_len_oos = 1000
    x_var_oos = gen_x_var(data_len_oos, std=1)
    error_oos = gen_error(data_len_oos, std=std_err)
    y_var_oos = func(x_var_oos, error_oos)
    y_predict_log, y_predict_with_var_raw_y, y_predict_with_var_err, y_predict_without_var = model.predict(x_var_oos)

    r_sq_without_var = model.report_r_sq(y_var_oos, y_predict_without_var)
    r_sq_with_var_raw = model.report_r_sq(y_var_oos, y_predict_with_var_raw_y)
    r_sq_with_var_err = model.report_r_sq(y_var_oos, y_predict_with_var_err)
    r_sq_log = model.report_r_sq(np.log(y_var_oos), y_predict_log)

    # y_predict_log, y_predict_with_var, y_predict_without_var = model.predict(x_var)
    #
    # r_sq_without_var = model.report_r_sq(y_var, y_predict_without_var)
    # r_sq_with_var = model.report_r_sq(y_var, y_predict_with_var)
    # r_sq_log = model.report_r_sq(np.log(y_var), y_predict_log)

    r_sq_in_sample = model.report_r_sq(y_real=np.log(y_var), y_predict=model.y_var_predict_log_in_sample)

    s = 'without: {}, with_raw: {}, with_err: {}, log: {}, in_sample: {}, in_sample2: {}'.format(
        r_sq_without_var, r_sq_with_var_raw, r_sq_with_var_err, r_sq_log, r_sq_in_sample, model.param_reg.rsquared
    )

    print(s)


class ModelLogNormal:
    def __init__(self, y_var, x_var):
        self.y_var = y_var
        self.x_var = x_var
        self.y_var_log = np.log(y_var)
        self.x_var_log = np.log(x_var)

        self.y_var_predict_log_in_sample = None
        self.error_in_sample = None

        self.model = None
        self.param_reg = None

    def fit(self):
        self.model = sm.OLS(endog=self.y_var_log, exog=sm.add_constant(self.x_var_log))
        self.param_reg = self.model.fit()
        self.y_var_predict_log_in_sample = self.predict_log(self.x_var_log)
        self.error_in_sample = self.y_var_log - self.y_var_predict_log_in_sample

    def predict_log(self, x_var_oos_log):
        exog = sm.add_constant(x_var_oos_log)
        y_predict_log = self.model.predict(params=self.param_reg.params, exog=exog)
        return y_predict_log

    def predict(self, x_var_oos):
        x_var_oos_log = np.log(x_var_oos)
        y_predict_log = self.predict_log(x_var_oos_log)
        y_predict_with_var_raw_y = np.exp(y_predict_log + 1 / 2 * (self.y_var_log.std() ** 2))
        y_predict_with_var_err = np.exp(y_predict_log + 1 / 2 * (self.error_in_sample.std() ** 2))
        y_predict_without_var = np.exp(y_predict_log)
        return y_predict_log, y_predict_with_var_raw_y, y_predict_with_var_err, y_predict_without_var

    @classmethod
    def report_r_sq(cls, y_real, y_predict):
        err = y_real - y_predict
        msr = (err * err).mean()
        err2 = y_real - y_real.mean()
        mse = (err2 * err2).mean()
        r_sq = 1 - msr / mse
        return r_sq


def gen_x_var(data_len, std):
    normal_data = np.random.normal(size=data_len, scale=std)
    exp_normal_data = np.exp(normal_data)
    return exp_normal_data


def gen_error(data_len, std):
    err = np.random.normal(scale=std, size=data_len)
    return err


def func(x_var, error, a=.8, b=.4):
    log_x = np.log(x_var)
    log_y = a * log_x + b + error
    y_exp = np.exp(log_y)
    return y_exp


if __name__ == '__main__':
    main()
