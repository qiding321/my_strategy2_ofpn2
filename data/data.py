# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import copy
import os
import re

import numpy as np
import pandas as pd

import data.reg_data
import log.log
import my_path.path
import paras.paras
import util.const
import util.util

my_log = log.log.log_order_flow_predict
log_error = log.log.log_error


class DataBase:
    def __init__(self, this_paras=None, has_data_df=False, data_df=None):

        assert isinstance(this_paras, (None.__class__, paras.paras.Paras))
        assert isinstance(has_data_df, bool)
        assert isinstance(data_df, (None.__class__, pd.DataFrame))

        self.source_data_path = my_path.path.data_source_path
        self.paras = this_paras

        self.raw_data_len = None
        self.truncated_len_dict = None
        self.drop_na_data_len = None

        self.date_begin = ''
        self.date_end = ''
        self.have_data_df = has_data_df
        self.data_df = data_df

    def init_data_from_csv(self):
        if not self.paras.x_vars_para.moving_average_list:
            data_df = self._get_data(data_path=self.source_data_path, date_begin=self.date_begin,
                                     date_end=self.date_end, begin_20_days_before=True)
            for var_moving_average in self.paras.x_vars_para.moving_average_list:
                data_df[var_moving_average] = self._get_one_col(var_moving_average)
        else:
            data_df = self._get_data(data_path=self.source_data_path, date_begin=self.date_begin, date_end=self.date_end)

        # fill na
        data_df_fill_na = self._fill_na(data_df=data_df)

        # drop up-limit or down-limit cases
        data_df_fill_na[(data_df_fill_na['bid1'] == 0) | (data_df_fill_na['ask1'] == 0)] = np.nan
        self.data_df = data_df_fill_na

    def generate_reg_data(self, normalize_funcs=None, reg_data_training=None):
        my_log.info('{} begin'.format('reg_data_training' if reg_data_training is None else 'reg_data_predicting'))
        normalize = self.paras.normalize
        divided_std = self.paras.divided_std

        time_scale_x = self.paras.time_scale_paras.time_scale_x
        time_scale_y = self.paras.time_scale_paras.time_scale_y
        time_scale_now = self.paras.time_scale_paras.time_freq

        # generate series to reg
        y_series, x_series = self._get_series_to_reg()

        # lag and normalize
        if normalize_funcs is None:
            x_series_not_normalize, y_series_not_normalize = self._get_useful_lag_series(x_series, y_series, time_scale_x, time_scale_y, time_scale_now)
            x_series_new, y_series_new, normalize_funcs = self._normalize(x_series_not_normalize, y_series_not_normalize, normalize=normalize,
                                                                          divided_std=divided_std)
            x_series_rename = x_series_new.rename(columns=dict([(name, name + '_x') for name in x_series_new]))
            y_series_rename = y_series_new.rename(columns=dict([(name, name + '_y') for name in y_series_new]))

            reg_data = data.reg_data.RegDataTraining(x_vars=x_series_rename, y_vars=y_series_rename,
                                                     x_vars_before_normalize=x_series_not_normalize, y_vars_before_normalize=y_series_not_normalize,
                                                     paras_config=self.paras, normalize_funcs=normalize_funcs)

        else:
            x_series_not_normalize, y_series_not_normalize = self._get_useful_lag_series(x_series, y_series, time_scale_x, time_scale_y, time_scale_now)
            x_series_new, y_series_new, normalize_funcs = self._normalize(x_series_not_normalize, y_series_not_normalize, normalize=normalize,
                                                                          divided_std=divided_std, predict_funcs=normalize_funcs)
            x_series_rename = x_series_new.rename(columns=dict([(name, name + '_x') for name in x_series_new]))
            y_series_rename = y_series_new.rename(columns=dict([(name, name + '_y') for name in y_series_new]))

            reg_data = data.reg_data.RegDataTest(x_vars=x_series_rename, y_vars=y_series_rename,
                                                 x_vars_before_normalize=x_series_not_normalize, y_vars_before_normalize=y_series_not_normalize,
                                                 paras_config=self.paras, normalize_funcs=normalize_funcs, reg_data_training=reg_data_training)

        my_log.info('{} end'.format('reg_data_training' if reg_data_training is None else 'reg_data_predicting'))
        return reg_data, normalize_funcs

    def report_description_stats(self, output_path, file_name):
        this_path = output_path
        if os.path.exists(this_path):
            pass
        else:
            os.makedirs(this_path)
        file_path = this_path + file_name
        s_ = ''
        s_ += 'raw_data_len,{raw_data_len}\nfinal_data_len,{final_data_len}\n'.format(raw_data_len=self.raw_data_len, final_data_len=self.drop_na_data_len)
        if self.truncated_len_dict is not None:
            for k, v in self.truncated_len_dict.items():
                s_ += str(k) + '_truncated,' + str(v) + '\n'
        with open(file_path, 'w') as f_out:
            f_out.write(s_)

    @classmethod
    def _get_data(cls, data_path, date_begin, date_end, begin_20_days_before=False):
        file_list = os.listdir(data_path)
        date_list = [x.split('.')[0] for x in file_list]

        if begin_20_days_before:
            date_begin_actual = [date_ for date_ in date_list if date_begin <= date_ <= date_end][0]
            date_begin_idx = date_list.index(date_begin_actual)
            date_begin_20day_before_idx = max(0, date_begin_idx - 20)
            date_begin_20day_before = date_list[date_begin_20day_before_idx]
            date_begin_ = date_begin_20day_before
        else:
            date_begin_ = date_begin
        date_list_useful = [date_ for date_ in date_list if date_begin_ <= date_ <= date_end]
        my_log.info('date range: {}, {}, {} trading days'.format(date_list_useful[0], date_list_useful[-1], len(date_list_useful)))
        path_list_useful = [data_path + date_ + '.csv' for date_ in date_list_useful]
        data_list = [pd.read_csv(path_, date_parser=util.util.str2date_ymdhms, parse_dates=['time']) for path_ in path_list_useful]
        data_df = pd.DataFrame(pd.concat(data_list, ignore_index=True)).set_index('time').sort_index()
        return data_df

    @classmethod
    def _fill_na(cls, data_df):
        col_list = []
        for col_name in data_df.columns:
            col_new = util.util.fill_na_method(data_df[col_name], col_name)
            col_list.append(col_new)
        data_df_ = pd.DataFrame(pd.concat(col_list, axis=1, keys=data_df.columns))
        return data_df_

    def _get_useful_lag_series(self, x_vars, y_vars, time_scale_x, time_scale_y, time_scale_now):  # todo  check
        window_x = util.util.get_windows(time_scale_long=time_scale_x, time_scale_short=time_scale_now)
        window_y = util.util.get_windows(time_scale_long=time_scale_y, time_scale_short=time_scale_now)
        assert window_y >= window_x > 0

        contemporaneous_cols = self.paras.x_vars_para.moving_average_list + self.paras.x_vars_para.intraday_pattern_list
        non_contemp_cols = [col_ for col_ in x_vars.columns if col_ not in contemporaneous_cols]

        x_vars_not_contemp = x_vars[non_contemp_cols]
        x_vars_contemp = x_vars[contemporaneous_cols]

        if non_contemp_cols:
            x_series_not_contemp = x_vars_not_contemp.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
                lambda x:
                x
                    .resample(time_scale_x, label='right')
                    .mean()
                    .select(util.util.is_in_market_open_time)
            )
        else:
            x_series_not_contemp = pd.DataFrame()

        # add lag term
        for col_ in self.paras.x_vars_para.lag_list:
            lag_num = int(re.search('(?<=lag)\d*', col_).group())
            assert lag_num >= 2
            x_series_not_contemp[col_] = x_series_not_contemp[col_].shift(lag_num - 1)

        # for contemporaneous x-vars
        if contemporaneous_cols:
            x_series_contemp = x_vars_contemp.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
                lambda x:
                x.rolling(window=window_y).mean().shift(-window_y)
                    .resample(time_scale_x, label='right').apply('last')
                    .select(util.util.is_in_market_open_time)
            )
        else:
            x_series_contemp = pd.DataFrame()
        x_series = pd.merge(x_series_contemp, x_series_not_contemp, left_index=True, right_index=True, how='outer')

        # truncate for x vars
        if self.paras.x_vars_para.truncate_list:  # todo
            truncated_var_df, _, truncated_len_dict0 = self._get_truncate_vars(vars_=x_series, var_names_to_truncate=self.paras.x_vars_para.truncate_list)
            x_series[self.paras.x_vars_para.truncate_list] = truncated_var_df

        # jump for x vars
        if self.paras.x_vars_para.jump_list:  # todo
            _, truncated_dummy_df, truncated_len_dict1 = self._get_truncate_vars(vars_=x_series, var_names_to_truncate=self.paras.x_vars_para.jump_list)
            x_series[self.paras.x_vars_para.jump_list] = truncated_dummy_df

        # log vars truncate
        for log_var_name in self.paras.x_vars_para.log_list:
            x_series[log_var_name] = self._take_log_and_truncate(x_series[log_var_name])

        # y series
        y_series = y_vars.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
            lambda x:
            x.rolling(window=window_y).mean().shift(-window_y)
                .resample(time_scale_x, label='right').apply('last')
                .select(util.util.is_in_market_open_time)
        )
        y_var_type = util.util.get_var_type(self.paras.y_vars.y_vars_list[0])
        if y_var_type == util.const.VAR_TYPE.truncate:
            truncated_var_df, _, truncated_len_dict_y = self._get_truncate_vars(vars_=y_series, var_names_to_truncate=self.paras.y_vars.y_vars_list)
            y_series = truncated_var_df
        elif y_var_type == util.const.VAR_TYPE.jump:
            _, truncated_dummy_df, truncated_len_dict_y = self._get_truncate_vars(vars_=y_series, var_names_to_truncate=self.paras.y_vars.y_vars_list)
            y_series = truncated_dummy_df
        elif y_var_type == util.const.VAR_TYPE.log:
            for col_name in y_series:
                y_series[col_name] = self._take_log_and_truncate(y_series[col_name])
        else:
            assert y_var_type == util.const.VAR_TYPE.normal
        x_series_drop_na, y_series_drop_na = self._dropna(x_series, y_series)

        return x_series_drop_na, y_series_drop_na

    @classmethod
    def _normalize(cls, x_series_drop_na, y_series_drop_na, normalize, divided_std, predict_funcs=None):
        if predict_funcs is None:
            if normalize:
                # normalize modified: divided by std or not
                if divided_std:
                    def x_series_normalize_func(x_):
                        return pd.DataFrame(
                            [(x_[col] - x_series_drop_na[col].mean()) if col != 'mid_px_ret_dummy' else x_[col] for col in list(set(x_.columns)) if
                             x_series_drop_na[col].std() != 0]
                        ).T

                    def y_series_normalize_func(y_):
                        return pd.DataFrame(
                            [(y_[col] - y_series_drop_na[col].mean()) if col != 'mid_px_ret_dummy' else y_[col] for col in y_]).T

                    def x_series_normalize_func_reverse(x_):
                        return pd.DataFrame(
                            [(x_[col] + x_series_drop_na[col].mean()) if col != 'mid_px_ret_dummy' else x_[col] for col in list(set(x_.columns)) if
                             x_series_drop_na[col].std() != 0]
                        ).T

                    assert len(y_series_drop_na.columns) == 1

                    def y_series_normalize_func_reverse(y_):
                        return pd.DataFrame(
                            [(y_[col] + y_series_drop_na.iloc[:, 0].mean()) if col != 'mid_px_ret_dummy' else y_[col] for col in y_]).T

                else:
                    # divide version
                    def x_series_normalize_func(x_):
                        return pd.DataFrame(
                            [(x_[col] - x_series_drop_na[col].mean()) / x_series_drop_na[col].std() if col != 'mid_px_ret_dummy' else x_[col] for col in
                             list(set(x_.columns)) if
                             x_series_drop_na[col].std() != 0]
                        ).T

                    def y_series_normalize_func(y_):
                        return pd.DataFrame(
                            [(y_[col] - y_series_drop_na[col].mean()) / y_series_drop_na[col].std() if col != 'mid_px_ret_dummy' else y_[col] for col in y_]).T

                    def x_series_normalize_func_reverse(x_):
                        return pd.DataFrame(
                            [(x_[col] + x_series_drop_na[col].mean()) * x_series_drop_na[col].std() if col != 'mid_px_ret_dummy' else x_[col] for col in
                             list(set(x_.columns)) if
                             x_series_drop_na[col].std() != 0]
                        ).T

                    assert len(y_series_drop_na.columns) == 1

                    def y_series_normalize_func_reverse(y_):
                        return pd.DataFrame(
                            [(y_[col] * y_series_drop_na.iloc[:, 0].std() + y_series_drop_na.iloc[:, 0].mean()) if col != 'mid_px_ret_dummy'
                             else y_[col] for col in y_]).T

            else:
                x_series_normalize_func, y_series_normalize_func = lambda x: x, lambda x: x
                x_series_normalize_func_reverse, y_series_normalize_func_reverse = lambda x: x, lambda x: x
        else:
            if normalize:
                x_series_normalize_func = predict_funcs['x_series_normalize_func']
                y_series_normalize_func = predict_funcs['y_series_normalize_func']
                x_series_normalize_func_reverse = predict_funcs['x_series_normalize_func_reverse']
                y_series_normalize_func_reverse = predict_funcs['y_series_normalize_func_reverse']
            else:
                x_series_normalize_func, y_series_normalize_func = lambda x: x, lambda x: x
                x_series_normalize_func_reverse, y_series_normalize_func_reverse = lambda x: x, lambda x: x

        x_new = x_series_normalize_func(x_series_drop_na)
        y_new = y_series_normalize_func(y_series_drop_na)
        predict_funcs = {
            'x_series_normalize_func'        : x_series_normalize_func,
            'y_series_normalize_func'        : y_series_normalize_func,
            'x_series_normalize_func_reverse': x_series_normalize_func_reverse,
            'y_series_normalize_func_reverse': y_series_normalize_func_reverse
        }
        return x_new, y_new, predict_funcs

    def _get_series_to_reg(self):

        time_freq = self.paras.time_scale_paras.time_freq

        # ================================== y vars ==================================
        y_vars_name = self.paras.y_vars.y_vars_list
        y_vars_raw = self._get_vars(y_vars_name)  # must make sure y_vars_raw is '3s' frequency

        # ================================== x vars ==================================
        x_vars_name = self.paras.x_vars_para.x_vars_list

        x_vars_raw = self._get_vars(x_vars_name)
        self.raw_data_len = len(x_vars_raw)

        # ================================= drop na ===================================
        x_vars_dropna, y_vars_dropna = self._dropna(x_vars_raw, y_vars_raw, to_log=True)
        self.drop_na_data_len = len(x_vars_dropna)
        return y_vars_dropna, x_vars_dropna

    @classmethod
    def _dropna(cls, x_vars, y_vars, to_log=True):
        data_merged = pd.DataFrame(pd.concat([x_vars, y_vars], keys=['x', 'y'], axis=1))
        data_merged_drop_na = data_merged.dropna()
        x_vars_dropna = data_merged_drop_na['x']
        y_vars_dropna = data_merged_drop_na['y']

        if to_log:
            log_func = my_log.info
        else:
            log_func = my_log.debug
        log_func('data_length_raw: {}\ndata_length_na: {}\ndata_length_dropna: {}'
                 .format(len(data_merged), len(data_merged) - len(data_merged_drop_na), len(data_merged_drop_na)))

        return x_vars_dropna, y_vars_dropna

    def _get_vars(self, vars_name):
        assert isinstance(vars_name, list)
        my_data = self._get_data_cols(vars_name)
        # data_freq = my_data.groupby(util.util.datetime2ymdstr, group_keys=False).apply(
        #     lambda x: x.resample(time_freq).apply('mean').select(util.util.is_in_market_open_time))  # why groupby? Because of days which are not trading days
        # return data_freq
        return my_data

    def _get_data_cols(self, vars_name):
        data_list = []
        for var_name in vars_name:
            data_list.append(self._get_one_col(var_name))
        data_df = pd.DataFrame(pd.concat(data_list, keys=vars_name, axis=1))
        return data_df

    def _get_one_col(self, var_name):
        my_log.debug(var_name)
        data_raw = self.data_df
        var_type = util.util.get_var_type(var_name)

        if var_name in data_raw.columns:
            data_new = data_raw[var_name]

        elif var_type == util.const.VAR_TYPE.normal:
            if var_name == 'spread':
                data_new = data_raw['ask1'] - data_raw['bid1']
                data_new[(data_new >= self.paras['spread_threshold'][1]) | (data_new <= self.paras['spread_threshold'][0])] = np.nan
            elif var_name == 'mid_px_ret':
                mid_px = (data_raw['ask1'] + data_raw['bid1']) / 2
                mid_px_ret = mid_px / (mid_px.shift(1)) - 1
                data_new = mid_px_ret
            elif var_name == 'mid_px_ret_dummy':
                mid_px = (data_raw['ask1'] + data_raw['bid1']) / 2
                mid_px_ret = mid_px / mid_px.shift(1) - 1
                data_new = pd.Series(np.where(mid_px_ret == 0, [1] * len(mid_px_ret), [0] * len(mid_px_ret)), index=mid_px.index)
            elif var_name == 'ret_sh50':
                sh50_px = data_raw['price_index_sh50']
                data_new = sh50_px / sh50_px.shift(1) - 1
            elif var_name == 'ret_index_index_future_300':
                index_future_px = data_raw['price_index_index_future_300']
                data_new = index_future_px / index_future_px.shift(1) - 1
            elif var_name == 'ret_hs300':
                sh300_px = data_raw['price_index_hs300']
                data_new = sh300_px / sh300_px.shift(1) - 1
            elif var_name == 'bid1_ret':
                bid1 = data_raw['bid1']
                data_new = bid1 / bid1.shift(1) - 1
            elif var_name == 'ask1_ret':
                ask1 = data_raw['ask1']
                data_new = ask1 / ask1.shift(1) - 1
            elif var_name == 'volatility_index300_60s':
                sh300_px = data_raw['price_index_hs300']
                sh300_ret = sh300_px.pct_change().fillna(method='ffill')
                data_new = sh300_ret.rolling(window=20).std()
            elif var_name == 'volatility_index50_60s':
                sh300_px = data_raw['price_index_sh50']
                sh300_ret = sh300_px.pct_change().fillna(method='ffill')
                data_new = sh300_ret.rolling(window=20).std()
            elif var_name == 'volatility_mid_px_60s':
                mid_px = (data_raw['ask1'] + data_raw['bid1']) / 2
                mid_px_ret = mid_px.pct_change().fillna(method='ffill')
                data_new = mid_px_ret.rolling(window=20).std()
            elif var_name == 'bsize1_change':
                data_new = data_raw['bsize1'] - data_raw['bsize1'].shift(1)
            elif var_name == 'asize1_change':
                data_new = data_raw['asize1'] - data_raw['asize1'].shift(1)
            elif var_name == 'buy_vol_10min_intraday_pattern_20_days':
                data_vol = data_raw[['buyvolume']]
                data_vol.loc[:, 'index'] = data_vol.index
                data_vol.loc[:, 'date'] = data_vol['index'].apply(lambda x: (x.year, x.month, x.day))
                data_vol.loc[:, 'period'] = data_vol['index'].apply(util.util.in_intraday_period)
                data_vol['new_index'] = list(zip(data_vol['date'], data_vol['period']))

                vol_mean_by_date_and_period = data_vol.groupby(['date', 'period'])['buyvolume'].mean()
                vol_wide = vol_mean_by_date_and_period.unstack().sort_index()
                vol_wide_rolling_mean = vol_wide.rolling(window=20).mean().shift(1)
                vol_long = vol_wide_rolling_mean.stack()

                data_new = pd.DataFrame(vol_long[data_vol['new_index']]).set_index(data_vol.index)[0]
            elif var_name == 'sell_vol_10min_intraday_pattern_20_days':
                data_vol = data_raw[['sellvolume']]
                data_vol.loc[:, 'index'] = data_vol.index
                data_vol.loc[:, 'date'] = data_vol['index'].apply(lambda x: (x.year, x.month, x.day))
                data_vol.loc[:, 'period'] = data_vol['index'].apply(util.util.in_intraday_period)
                data_vol['new_index'] = list(zip(data_vol['date'], data_vol['period']))

                vol_mean_by_date_and_period = data_vol.groupby(['date', 'period'])['sellvolume'].mean()
                vol_wide = vol_mean_by_date_and_period.unstack().sort_index()
                vol_wide_rolling_mean = vol_wide.rolling(window=20).mean().shift(1)
                vol_long = vol_wide_rolling_mean.stack()

                data_new = pd.DataFrame(vol_long[data_vol['new_index']]).set_index(data_vol.index)[0]
            else:
                my_log.error(var_name)
                raise LookupError
        # moving average terms
        elif var_type == util.const.VAR_TYPE.moving_average:
            var_name_prefix = '_'.join(var_name.split('_')[:-1])
            ma_days = int(re.search('(?<=mean)\d+(?=day)', var_name).group())
            data_col = data_raw[[var_name_prefix]]
            data_col.loc[:, 'index'] = data_col.index
            data_col.loc[:, 'date'] = data_col['index'].apply(lambda x: (x.year, x.month, x.day))
            data_col['new_index'] = data_col['date']

            data_mean_by_date_and_period = data_col.groupby(['date'])[var_name_prefix].mean()
            data_wide = data_mean_by_date_and_period.sort_index()
            data_wide_rolling_mean = data_wide.rolling(window=ma_days).mean().shift(1)
            data_long = data_wide_rolling_mean
            data_new = pd.DataFrame(data_long[data_col['new_index']]).set_index(data_col.index)

        elif var_type == util.const.VAR_TYPE.high_order:
            var_name_prefix = var_name[:-7]
            data_new_ = self._get_one_col(var_name_prefix)
            # data_new = data_new_.values * data_new_.values
            # data_new = pd.DataFrame(data_new, index=data_new_.index)
            data_new = data_new_ * data_new_
        elif var_type == util.const.VAR_TYPE.truncate:
            var_name_new = var_name.replace('_truncate', '')
            data_new = self._get_one_col(var_name_new)
        elif var_type == util.const.VAR_TYPE.log:
            var_name_new = var_name.replace('_log', '')
            # data_new_ = copy.deepcopy(self._get_one_col(var_name=var_name_new))
            # zero_idx = data_new_ <= 0
            # data_new_[zero_idx] = np.nan
            # data_new = np.log(data_new_)
            # data_new[zero_idx] = -np.inf
            data_new = self._get_one_col(var_name_new)
        elif var_type == util.const.VAR_TYPE.lag:
            var_name_new = re.search('.*(?=_lag\d*)', var_name).group()
            data_new = self._get_one_col(var_name_new)
        else:
            my_log.error(var_name)
            my_log.error(var_type)
            raise LookupError

        return copy.deepcopy(data_new)

    def _get_truncate_vars(self, vars_, var_names_to_truncate):
        truncate_para = self.paras.truncate_paras
        method_ = truncate_para.truncate_method
        truncate_window = truncate_para.truncate_window
        truncate_std = truncate_para.truncate_std
        truncated_dummy_list = []
        truncated_var_list = []
        if method_ == 'mean_std':
            for var_ in var_names_to_truncate:
                my_log.info('truncate begin: ' + var_)
                truncated_var_, truncated_dummy_ = self._truncate_mean_std(var_col=vars_[var_], window=truncate_window,
                                                                           truncate_std=truncate_std)
                truncated_var_list.append(truncated_var_)
                truncated_dummy_list.append(truncated_dummy_)
                my_log.info('truncate end: {}, truncated num: {}'.format(var_, len(truncated_dummy_[truncated_dummy_ != 0])))
        else:
            my_log.error('wrong truncate method: {}'.format(method_))
            raise ValueError
        truncated_dummy_df = pd.concat(truncated_dummy_list, axis=1, keys=var_names_to_truncate) if truncated_dummy_list else None
        truncated_var_df = pd.concat(truncated_var_list, axis=1, keys=var_names_to_truncate) if truncated_var_list else None
        truncated_len_dict = dict(list(zip(var_names_to_truncate, [len(t_d_[t_d_ != 0]) for t_d_ in truncated_dummy_list])))

        return truncated_var_df, truncated_dummy_df, truncated_len_dict

    @classmethod
    def _truncate_mean_std(cls, var_col, window, truncate_std):
        n_ = len(var_col)
        var_col_new = pd.Series([np.nan] * n_, index=var_col.index)
        var_col_dummy = pd.Series([np.nan] * n_, index=var_col.index)
        var_col_new.iloc[0:window] = var_col.iloc[0:window]
        for i in range(window, n_):
            var_tmp = var_col_new[i - window:i]
            mean_tmp = var_tmp.mean()
            std_tmp = var_tmp.std()
            point_raw = var_col.iloc[i]
            if point_raw >= mean_tmp + truncate_std * std_tmp:
                point_new = mean_tmp + truncate_std * std_tmp
                dummy = 1
            # elif point_raw <= mean_tmp - truncate_std * std_tmp:
            #     point_new = mean_tmp - truncate_std * std_tmp
            #     dummy = -1
            else:
                point_new = point_raw
                dummy = 0
            var_col_new.iloc[i] = point_new
            var_col_dummy.iloc[i] = dummy
        return var_col_new, var_col_dummy

    @classmethod
    def _take_log_and_truncate(cls, var_col):
        zero_index = var_col <= 0
        var_col[zero_index] = np.nan
        min_ = var_col.min()
        var_col[zero_index] = min_
        var_col_log = np.log(var_col)
        return var_col_log


class TrainingData(DataBase):
    def __init__(self, this_paras=None, has_data_df=False, data_df=None):
        DataBase.__init__(self, this_paras=this_paras, has_data_df=has_data_df, data_df=data_df)
        if not has_data_df:
            self.date_begin = self.paras.period_paras.begin_date_training
            self.date_end = self.paras.period_paras.end_date_training
            self.init_data_from_csv()
        else:
            data_df = data_df.sort_index()
            self.date_begin = data_df.index[0]
            self.date_end = data_df.index[-1]
            self.data_df = data_df


class TestingData(DataBase):
    def __init__(self, this_paras=None, has_data_df=False, data_df=None):
        DataBase.__init__(self, this_paras=this_paras, has_data_df=has_data_df, data_df=data_df)
        if not has_data_df:
            self.date_begin = self.paras.period_paras.begin_date_predict
            self.date_end = self.paras.period_paras.end_date_predict
            self.init_data_from_csv()
        else:
            data_df = data_df.sort_index()
            self.date_begin = data_df.index[0]
            self.date_end = data_df.index[-1]
            self.data_df = data_df
