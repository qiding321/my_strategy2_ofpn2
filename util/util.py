# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:47

@author: qiding
"""

import datetime
import pickle
import re

import pandas as pd

import log.log
import util.const

my_log = log.log.log_order_flow_predict


def get_timenow_str():
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d-%H-%M-%S')
    return now_str


def str2date_ymdhms(date_str):
    date_datetime = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return date_datetime


def is_in_market_open_time(time):
    time_new = datetime.datetime(1900, 1, 1, time.hour, time.minute, time.second)
    b = util.const.MARKET_OPEN_TIME <= time_new <= util.const.MARKET_CLOSE_TIME_NOON or util.const.MARKET_OPEN_TIME_NOON <= time_new <= \
                                                                                        util.const.MARKET_END_TIME
    return b


def datetime2ymdstr(time):
    s = time.strftime('%Y-%m-%d')
    return s


def get_var_type(var_name):
    if var_name in [
        'buyvolume_mean5days', 'buyvolume_mean20days', 'buyvolume_mean1day',
        'sellvolume_mean5days', 'sellvolume_mean20days', 'sellvolume_mean1day',
        'volume_index_sh50_mean5days', 'volume_index_sh50_mean20days', 'volume_index_sh50_mean1day',
        'volume_index_hs300_mean5days', 'volume_index_hs300_mean20days', 'volume_index_hs300_mean1day',
    ]:
        var_type = util.const.VAR_TYPE.moving_average
    elif var_name.find('_order2') >= 0:
        var_type = util.const.VAR_TYPE.high_order
    elif var_name.find('_truncate') >= 0:
        var_type = util.const.VAR_TYPE.truncate
    elif (var_name.find('_jump') >= 0) and (var_name.find('_jump_freq') < 0):
        var_type = util.const.VAR_TYPE.jump
    elif var_name.find('_jump_freq') >= 0:
        var_type = util.const.VAR_TYPE.jump_freq
    elif (var_name.find('_log') >= 0) and (var_name.find('_change') < 0):
        var_type = util.const.VAR_TYPE.log
    elif re.search('.*(?=_lag\d)', var_name) is not None:
        var_type = util.const.VAR_TYPE.lag
    elif (var_name.find('_log') >= 0) and (var_name.find('_change') >= 0):
        var_type = util.const.VAR_TYPE.log_change
    elif var_name.find('_abs') >= 0:
        var_type = util.const.VAR_TYPE.abs
    elif var_name.find('_cross') >= 0:
        var_type = util.const.VAR_TYPE.cross_term
    else:
        var_type = util.const.VAR_TYPE.normal

    return var_type


def get_windows(time_scale_long, time_scale_short='3s'):
    td0 = pd.datetools.to_offset(time_scale_short).delta
    td1 = pd.datetools.to_offset(time_scale_long).delta
    windows = int(td1 / td0)
    return windows


def fill_na_method(data_series, col_name):
    if col_name.find('date') >= 0:
        data_series_new = data_series.fillna(method='ffill')
    elif col_name.find('price') >= 0:
        data_series_new = data_series.fillna(method='ffill')
    elif col_name.find('ret') >= 0:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('vol') >= 0 and col_name.find('accvolume') == -1:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('accvolume') >= 0:
        data_series_new = data_series.fillna(method='ffill')
    elif any([col_name.find('bid') >= 0, col_name.find('ask') >= 0, col_name.find('bsize') >= 0, col_name.find('asize') >= 0]):
        data_series_new = data_series.fillna(method='ffill')
    elif col_name.find('amount') >= 0:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('trans') >= 0:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('change') >= 0:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('asize') >= 0 or col_name.find('bsize') >= 0:
        data_series_new = data_series.fillna(value='ffill')
    else:
        my_log.error('col_name: {}'.format(col_name))
        raise LookupError
    return data_series_new


def dump_pkl(obj_, file_path):
    with open(file_path, 'wb') as f_out:
        pickle.dump(obj_, f_out)
    my_log.info('pickle dump done: {}'.format(file_path))


def load_pkl(file_path):
    with open(file_path, 'rb') as f_in:
        data = pickle.load(f_in)
    my_log.info('pickle loading done: {}'.format(file_path))
    return data


def in_intraday_period(time, time_period='10min'):
    time_ = datetime.datetime(1900, 1, 1, hour=time.hour, minute=time.minute, second=time.second)
    seconds = get_seconds(end_time=time_)
    # td0 = datetime.timedelta(seconds=seconds)
    # td1 = datetime.timedelta(seconds=600)
    if time_period == '10min':
        time_period_seconds = 600
    else:
        log.log.log_price_predict.error('time_period_error: {}'.format(time_period))
        raise ValueError
    period = int(seconds / time_period_seconds)
    return period


def get_seconds(start_time=util.const.MARKET_OPEN_TIME, end_time=util.const.MARKET_END_TIME):
    seconds = (end_time - start_time).total_seconds()
    if end_time >= util.const.MARKET_OPEN_TIME_NOON and start_time <= util.const.MARKET_CLOSE_TIME_NOON:
        seconds -= 1.5 * 3600
    return seconds


def winsorize(series, quantile=(0.01, 0.99)):
    quantile0 = quantile[0]
    quantile1 = quantile[1]
    q0 = series.quantile(quantile0)
    q1 = series.quantile(quantile1)
    idx_bool = (series >= q0) & (series <= q1)
    s_ = series[(series >= q0) & (series <= q1)]
    return s_, idx_bool


def get_offset(time_period):
    if time_period == '12M':
        offset = pd.tseries.offsets.MonthEnd(12)
    elif time_period == '24M':
        offset = pd.tseries.offsets.MonthEnd(24)
    elif time_period == '1M':
        offset = pd.tseries.offsets.MonthEnd(1)
    elif time_period == '3M':
        offset = pd.tseries.offsets.MonthEnd(3)
    elif time_period == '6M':
        offset = pd.tseries.offsets.MonthEnd(6)
    elif time_period == '1D':
        offset = pd.tseries.offsets.Day(1)
    else:
        raise ValueError

    return offset


def resample_to_index(data_source, idx_destn, funcs):  # todo, check
    data_dict = dict()
    for col in data_source.columns:
        idx_source_iter = data_source.index.__iter__()
        idx_destn_iter = idx_destn.__iter__()

        idx_source_ = idx_source_iter.__next__()
        idx_destn_ = idx_destn_iter.__next__()

        this_col = data_source[col]
        this_func = util.const.func_mapping[funcs[col]]
        this_col_ = dict()
        while True:
            idx_source_data_tmp = []

            while True:
                if idx_source_ > idx_destn_:
                    break
                else:
                    idx_source_data_tmp.append(this_col[idx_source_])
                    try:
                        idx_source_ = idx_source_iter.__next__()
                    except StopIteration:
                        break
            this_data = this_func(idx_source_data_tmp)
            this_col_[idx_destn_] = this_data
            try:
                idx_destn_ = idx_destn_iter.__next__()
            except StopIteration:
                break

        data_dict[col] = pd.Series(this_col_, index=idx_destn)

    data_df = pd.DataFrame(data_dict).reindex(index=idx_destn, columns=data_source.columns)

    return data_df




if __name__ == '__main__':
    import paras.paras
    import util.util
    # import pandas as pd
    para = paras.paras.Paras()
    x_series = para.x_vars_para.x_vars_list
    for col_name in x_series:
        util.util.fill_na_method(pd.DataFrame(), col_name)