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
    elif var_name.find('_jump') >= 0:
        var_type = util.const.VAR_TYPE.jump
    elif (var_name.find('_log') >= 0) and (var_name.find('_change') < 0):
        var_type = util.const.VAR_TYPE.log
    elif re.search('.*(?=_lag\d)', var_name) is not None:
        var_type = util.const.VAR_TYPE.lag
    elif (var_name.find('_log') >= 0) and (var_name.find('_change') >= 0):
        var_type = util.const.VAR_TYPE.log_change
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
    elif col_name.find('volume') >= 0 and col_name.find('accvolume') == -1:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('accvolume') >= 0:
        data_series_new = data_series.fillna(method='ffill')
    elif any([col_name.find('bid') >= 0, col_name.find('ask') >= 0, col_name.find('bsize') >= 0, col_name.find('asize') >= 0]):
        data_series_new = data_series.fillna(method='ffill')
    elif col_name.find('amount') >= 0:
        data_series_new = data_series.fillna(value=0)
    elif col_name.find('trans') >= 0:
        data_series_new = data_series.fillna(value=0)
    else:
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


def winsorize(series, quantile):
    quantile0 = quantile[0]
    quantile1 = quantile[1]
    q0 = series.quantile(quantile0)
    q1 = series.quantile(quantile1)
    idx_bool = (series >= q0) & (series <= q1)
    s_ = series[(series >= q0) & (series <= q1)]
    return s_, idx_bool
