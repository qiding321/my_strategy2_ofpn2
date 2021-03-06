# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import data.data
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

        self.var_predict = None

    def predict(self):
        y_predict = self.model.predict(exog_new=self.x_vars, endg_new=self.y_vars)
        self.y_predict = y_predict
        normalize_funcs_reverse = self.normalize_funcs['y_series_normalize_func_reverse']
        self.y_predict_before_normalize = normalize_funcs_reverse(self.y_predict)
        if self.paras_config.method_paras.method == util.const.FITTING_METHOD.GARCH:
            self.var_predict = self.model.predict_var()

    def report_summary(self, output_path, file_name):
        summary = self.model.summary()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(output_path + file_name, 'w') as f_out:
            f_out.write(summary)

    def report_risk_analysis(self, output_path, file_name, bars=None, percent_num=40):
        if self.paras_config.method_paras.method == util.const.FITTING_METHOD.GARCH:
            var_predict = self.var_predict
        elif self.paras_config.method_paras.method in \
                [util.const.FITTING_METHOD.LOGIT, util.const.FITTING_METHOD.PROBIT, util.const.FITTING_METHOD.DECTREE]:
            var_predict = self.y_predict
        else:
            my_log.info('no var to analysis')
            return None, None, None, None
        y_raw = self.y_vars_raw
        if self.paras_config.method_paras.method == util.const.FITTING_METHOD.GARCH:
            _, truncated_dummy_df, truncated_len_dict_y = data.data.DataBase.get_truncate_vars(
                vars_=y_raw, var_names_to_truncate=y_raw.columns, truncate_para=self.paras_config.truncate_paras
            )
            y_raw = truncated_dummy_df
        assert var_predict is not None
        assert len(y_raw) == len(var_predict)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            my_log.info('make dirs: {}'.format(output_path))

        if self.paras_config.method_paras.method != util.const.FITTING_METHOD.DECTREE:
            # quantile
            if bars is None:
                percent = np.arange(0, 1, 1 / percent_num)
                percentile = [np.percentile(var_predict, y_ * 100) for y_ in percent]
                percentile.append(var_predict.max())
            else:
                percent_num = len(bars) - 1
                percentile = bars
            df_list = []
            ratio_list = []
            predict_value_list = []
            total_num_list = []
            target_num_list = []
            for idx_ in range(percent_num):
                lower_bar = percentile[idx_]
                higher_bar = percentile[idx_ + 1]
                idx_local = (var_predict < higher_bar) & (var_predict >= lower_bar)
                data_local = y_raw[idx_local]
                len_0 = len(data_local)
                len_1 = len(data_local[data_local == 1].dropna())
                if len_0 == 0:
                    my_log.warning('dvd by zero, {}'.format(output_path))
                    ratio = 0
                else:
                    ratio = len_1 / len_0
                predict_value = var_predict[idx_local].mean()

                target_num_list.append(len_1)
                total_num_list.append(len_0)
                df_list.append([len_0, len_1])
                ratio_list.append(ratio)
                predict_value_list.append(predict_value)
            # df = pd.DataFrame(df_list, index=['{:.4f}'.format(pct_) for pct_ in percentile[0:-1]], columns=[0, 1])
            # df.plot(kind='bar')
            # plt.savefig(output_path + file_name + 'percentile.jpg')
            # plt.close()
            s_ = pd.Series(ratio_list, index=['{:.4f}'.format(pct_) for pct_ in percentile[0:-1]])
            plt.figure(figsize=[15, 20])
            ax = s_.plot(kind='bar')
            ax.set_xticklabels(['{:.2f},{},{}'.format(pv_, tn1, tn2)
                                for pv_, tn1, tn2 in zip(predict_value_list, target_num_list, total_num_list)])
            plt.grid()
            plt.savefig(output_path + file_name + 'percentile.jpg')
            plt.close()

            title_txt = ','.join(['pct_low', 'pct_high', 'mean', 'accuracy', 'target_num', 'all_num']) + '\n'
            s_out_list = list(zip(percentile[:-1], percentile[1:],
                                  predict_value_list, ratio_list, target_num_list, total_num_list))
            s_out = '\n'.join(list(','.join([str(x_) for x_ in x]) for x in s_out_list))
            with open(output_path + file_name + 'percentile.csv', 'w') as f_out:
                f_out.write(title_txt)
                f_out.write(s_out)

            return percentile, ratio_list[-1], target_num_list[-1], total_num_list[-1]
            # # abs value
            # percent = list(np.arange(var_predict.min(), var_predict.max(),
            #                          1 / percent_num * (var_predict.max() - var_predict.min())))
            # percent.append(var_predict.max())
            # df_list = []
            # ratio_list = []
            # for idx_ in range(percent_num):
            #     lower_bar = percent[idx_]
            #     higher_bar = percent[idx_ + 1]
            #     idx_local = (var_predict < higher_bar) & (var_predict >= lower_bar)
            #     data_local = y_raw[idx_local]
            #     len_0 = len(data_local[data_local == 0].dropna())
            #     len_1 = len(data_local[data_local == 1].dropna())
            #     ratio = len_1 / (len_0 + len_1) if (len_0 + len_1) != 0 else 0
            #     df_list.append([len_0, len_1])
            #     ratio_list.append(ratio)
            # s_ = pd.Series(ratio_list, index=['{:.4f}'.format(pct_) for pct_ in percent[0:-1]])
            # ax = s_.plot(kind='bar')
            # title = ', '.join(['{}/{}'.format(len_0, len_1) for len_0, len_1 in df_list])
            # ax.set_title(title)
            # plt.savefig(output_path + file_name + 'abs_value_decile.jpg')
            # plt.close()
        else:
            predict_0 = var_predict == 0
            predict_1 = var_predict == 1
            raw_0 = (y_raw == 0).T.values
            raw_1 = (y_raw == 1).T.values
            _get_len = lambda raw_bool, predict_bool: (raw_bool & predict_bool).sum()
            len_0_0 = _get_len(predict_0, raw_0)
            len_0_1 = _get_len(predict_0, raw_1)
            len_1_0 = _get_len(predict_1, raw_0)
            len_1_1 = _get_len(predict_1, raw_1)
            df = pd.DataFrame([[len_0_0, len_0_1], [len_1_0, len_1_1]],
                              columns=['y_raw_0', 'y_raw_1'], index=['predict_0', 'predict_1'])
            ax = df.plot(kind='bar')
            plt.savefig(output_path + file_name + 'percentile.jpg')
            plt.close()
            return None, None, None, None

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
        r_squared_daily.to_csv(output_path + file_name[0])

        plt.plot(r_squared_daily['rsquared'][0].values)
        plt.savefig(output_path + file_name[1])
        plt.close()

    def plot_daily_fitting(self, output_path):
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
            my_log.info('make dirs: {}'.format(output_path))

        data_merged = self._get_y_predict_merged()
        for key, data_one_day in data_merged.groupby('ymd'):
            if 'y_var_predict' not in data_one_day.columns:
                fig = plt.figure()
                plt.plot(data_one_day['y_raw'].values, 'r-', label='y_raw')
                plt.plot(data_one_day['y_predict'].values, 'b-', label='y_predict')
                plt.legend(fontsize='small')
                fig.savefig(output_path + 'predict_volume_vs_raw_volume' + '-'.join([str(k_) for k_ in key]) + '.jpg')
                plt.close()
            else:
                fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
                ax0.plot(data_one_day['y_raw'].values, 'r-', label='y_raw')
                ax0.plot(data_one_day['y_predict'].values, 'b-', label='y_predict')
                ax0.legend(fontsize='small')
                ax0.set_title('y_raw and y_predict')
                ax1.plot(data_one_day['y_var_predict'].values)
                ax1.set_title('var_predict')
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
        plt.title(file_name)
        plt.savefig(output_path + file_name + '.jpg')
        plt.close()

    def report_corr_matrix(self, output_path, file_name):
        try:
            self.x_vars_raw.corr().to_csv(output_path + file_name)
        except Exception as e:
            my_log.error('corr_matrix error: ' + str(e))

    def report_variance(self, output_path, file_name):
        try:
            self.x_vars_raw.var().to_csv(output_path + file_name)
        except Exception as e:
            my_log.error('corr_matrix error: ' + str(e))

    def predict_y_hist(self, output_path, file_name):
        y_predict = self.y_predict_before_normalize
        plt.hist(y_predict, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.title(file_name)
        plt.savefig(output_path + file_name + '.jpg')
        plt.close()

    def record_error_description(self, output_path, file_name):
        data_merged = self._get_y_predict_merged()
        error_this_month = data_merged['error']

        err_des = error_this_month.describe()
        err_des['skew'] = error_this_month.skew()
        err_des['kurt'] = error_this_month.kurt()

        adf, pvalue, usedlag, nobs, crit_values, icbest = adfuller(error_this_month)
        err_des['adf_pvalue'] = pvalue
        err_des['used_lag'] = usedlag

        err_des.to_csv(output_path + file_name)

    def plot_y_var_hist(self, output_path, file_name):
        y_var = self.y_vars_raw.values
        plt.hist(y_var, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.title(file_name)
        plt.savefig(output_path + file_name + '.jpg')
        plt.close()

    def plot_x_var_hist(self, output_path):
        x_vars = self.x_vars_raw
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
        for col_name, x_var in x_vars.iteritems():
            plt.hist(x_var.values, 100, facecolor='b')
            plt.axvline(0, color='red')
            plt.title(col_name)
            plt.savefig(output_path + col_name + '.jpg')
            plt.close()

    def report_resume_if_logged(self, output_path):
        y_var_name = self.paras_config.y_vars.y_vars_list[0]
        y_var_type = util.util.get_var_type(y_var_name)
        if y_var_type == util.const.VAR_TYPE.log:
            if os.path.exists(output_path):
                pass
            else:
                os.makedirs(output_path)
                my_log.info('make dirs: {}'.format(output_path))
        else:
            return

        y_raw = np.exp(self.y_vars_raw)
        y_training = np.exp(self.reg_data_training.y_vars_raw.values.T[0])
        y_log_err_std_estimate = (self.reg_data_training.y_vars_raw.values.T[0] - self.reg_data_training.y_predict_before_normalize).std()
        y_predict = np.exp(pd.DataFrame(self.y_predict_before_normalize, index=y_raw.index, columns=['y_predict'])) * np.exp(y_log_err_std_estimate**2/2)
        data_merged = pd.merge(y_raw, y_predict, left_index=True, right_index=True).rename(
            columns={y_raw.columns[0]: 'y_raw', y_predict.columns[0]: 'y_predict'})
        data_merged['ymd'] = list(map(lambda x: (x.year, x.month, x.day), data_merged.index))
        data_merged['error'] = data_merged['y_raw'] - data_merged['y_predict']
        data_merged['sse'] = data_merged['y_raw'] - y_training.mean()

        def _generate_one_day_stats(c):
            mse_ = (c['sse'] * c['sse']).sum()
            msr_ = (c['error'] * c['error']).sum()
            r_sq_ = 1 - msr_ / mse_
            ret_ = pd.DataFrame([mse_, msr_, r_sq_], index=['mse', 'msr', 'rsquared']).T
            return ret_

        # report daily r-squared
        r_squared_daily = data_merged.groupby('ymd').apply(_generate_one_day_stats).unstack()
        r_squared_daily.to_csv(output_path + 'daily_r_squared.csv')
        plt.plot(util.util.winsorize(r_squared_daily['rsquared'][0], (0.01, 0.99))[0].values)
        plt.savefig(output_path + 'daily_r_squared.jpg')
        plt.close()

        error_this_month = data_merged['error']
        error_winsorized, idx_bool = util.util.winsorize(error_this_month, [0.01, 0.99])
        plt.hist(error_winsorized.values, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.savefig(output_path + 'error_hist.jpg')
        plt.close()

        plt.scatter(data_merged[idx_bool]['y_raw'], data_merged[idx_bool]['y_predict'], color='b')
        minmin = min(data_merged[idx_bool]['y_raw'].min(), data_merged[idx_bool]['y_predict'].min())
        maxmax = max(data_merged[idx_bool]['y_raw'].max(), data_merged[idx_bool]['y_predict'].max())
        plt.plot([minmin, maxmax], [minmin, maxmax], 'r-')
        plt.xlabel('y_raw')
        plt.ylabel('y_predict')
        plt.savefig(output_path + 'scatter.jpg')
        plt.close()

        plt.plot(data_merged[idx_bool]['y_raw'].values, color='b')
        plt.plot(data_merged[idx_bool]['y_predict'].values, color='r')
        plt.savefig(output_path + 'time_series.jpg')
        plt.close()

        err_des = error_this_month.describe()
        err_des['skew'] = error_this_month.skew()
        err_des['kurt'] = error_this_month.kurt()
        err_des.to_csv(output_path + 'error_stats.csv')

        plt.hist(y_raw.values, 100, facecolor='b')
        plt.axvline(0, color='red')
        plt.savefig(output_path + 'y_testing_hist.jpg')
        plt.close()

        # report daily fiiting
        if os.path.exists(output_path+'daily_fitting\\'):
            pass
        else:
            os.makedirs(output_path+'daily_fitting\\')
            my_log.info('make dirs: {}'.format(output_path+'daily_fitting\\'))

        for key, data_one_day in data_merged.groupby('ymd'):
            fig = plt.figure()
            plt.plot(data_one_day['y_raw'].values, 'r-', label='y_raw')
            plt.plot(data_one_day['y_predict'].values, 'b-', label='y_predict')
            plt.legend(fontsize='small')
            fig.savefig(output_path+'daily_fitting\\' + 'predict_volume_vs_raw_volume' + '-'.join([str(k_) for k_ in key]) + '.jpg')
            plt.close()
            fig = plt.figure()

            plt.scatter(data_one_day['y_raw'], data_one_day['y_predict'], color='b')
            minmin = min(data_one_day['y_raw'].min(), data_one_day['y_predict'].min())
            maxmax = max(data_one_day['y_raw'].max(), data_one_day['y_predict'].max())
            plt.plot([minmin, maxmax], [minmin, maxmax], 'r-')
            plt.xlabel('y_raw')
            plt.ylabel('y_predict')
            fig.savefig(output_path+'daily_fitting\\' + 'scatter' + '-'.join([str(k_) for k_ in key]) + '.jpg')
            plt.close()

    def get_r_squared(self):
        data_merged = self._get_y_predict_merged()
        error = data_merged['error']
        sse = data_merged['sse']
        r = 1 - (error * error).sum() / (sse * sse).sum()
        return r

    def _get_y_predict_merged(self):
        y_raw = self.y_vars_raw
        y_training = self.reg_data_training.y_vars_raw.values.T[0]
        y_predict = pd.DataFrame(self.y_predict_before_normalize, index=y_raw.index, columns=['y_predict'])
        data_merged = pd.merge(y_raw, y_predict, left_index=True, right_index=True).rename(
            columns={y_raw.columns[0]: 'y_raw', y_predict.columns[0]: 'y_predict'})
        data_merged['ymd'] = list(map(lambda x: (x.year, x.month, x.day), data_merged.index))
        data_merged['error'] = data_merged['y_raw'] - data_merged['y_predict']
        data_merged['sse'] = data_merged['y_raw'] - y_training.mean()

        if self.var_predict is not None:
            y_var_predict = pd.DataFrame(self.var_predict, index=y_raw.index, columns=['y_var_predict'])
            data_merged = pd.merge(data_merged, y_var_predict, left_index=True, right_index=True)

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
        cov_y_y_predict_multiplied_by_minus_2 = -2 * np.cov([y_actual, y_predict])[0, 1]

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
        elif method.method == util.const.FITTING_METHOD.LOGIT:
            self.model = method_wrapper.reg_method_wrapper.LogitWrapper(endog=self.y_vars, exog=self.x_vars,
                                                                        has_const=add_const)
            self.paras_reg = self.model.fit()
            self.predict()
        elif method.method == util.const.FITTING_METHOD.PROBIT:
            self.model = method_wrapper.reg_method_wrapper.ProbitWrapper(endog=self.y_vars, exog=self.x_vars,
                                                                         has_const=add_const)
            self.paras_reg = self.model.fit()
            self.predict()
        elif method.method == util.const.FITTING_METHOD.GARCH:
            self.model = method_wrapper.reg_method_wrapper.GarchWrapper(endog=self.y_vars, exog=self.x_vars)
            self.paras_reg = self.model.fit()
            # y_predict_insample = self.model.predict(exog_new=self.x_vars, endg_new=self.y_vars)
            # self.y_predict = y_predict_insample
            self.predict()
        elif method.method == util.const.FITTING_METHOD.DECTREE:
            self.model = method_wrapper.reg_method_wrapper.DecisionTreeWrapper(endog=self.y_vars, exog=self.x_vars,
                                                                               para=self.paras_config.decision_tree_paras)
            self.paras_reg = self.model.fit()
            self.predict()
            # self.model = DecisionTreeClassifier(max_depth=decision_tree_depth)
            # self.model.fit(self.x_vars, self.y_vars)
            # y_predict_insample = self.model.predict(self.x_vars)
            # self.y_predict = y_predict_insample
        # elif method.method == util.const.FITTING_METHOD.DECTREEREG:
        #     decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
        #     self.model = DecisionTreeRegressor(max_depth=decision_tree_depth)
        #     self.model.fit(self.x_vars, self.y_vars)
        #     y_predict_insample = self.model.predict(self.x_vars)
        #     self.y_predict = y_predict_insample
        # elif method.method == util.const.FITTING_METHOD.ADABOOST:
        #     decision_tree_depth = self.paras_config.decision_tree_paras.decision_tree_depth
        #     rng = np.random.RandomState(1)
        #     self.model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=decision_tree_depth), n_estimators=300, random_state=rng)
        #     self.model.fit(self.x_vars, self.y_vars)
        #     y_predict_insample = self.model.predict(self.x_vars)
        #     self.y_predict = y_predict_insample
        else:
            my_log.error('reg_method not found: {}'.format(method))
            raise ValueError
        return self.paras_reg

    def save_reg_paras_to_pickle(self, pickle_path):
        with open(pickle_path, 'wb') as f_out:
            pickle.dump(self.paras_reg, f_out)
            my_log.info('pickle dump done: {}'.format(pickle_path))

    def save_reg_paras_to_csv(self, csv_path):
        # self.paras_reg.para[0]
        # with open(csv_path, 'w') as f_out:
        #     f_out.write(s)
        # my_log.info('csv done: {}'.format(csv_path))
        pass


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
