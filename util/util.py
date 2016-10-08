# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:47

@author: qiding
"""

import datetime
import re

import pandas as pd

import util.const


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
    elif var_name.endswith('order2'):
        var_type = util.const.VAR_TYPE.high_order
    elif var_name.endswith('truncate'):
        var_type = util.const.VAR_TYPE.truncate
    elif var_name.endswith('jump'):
        var_type = util.const.VAR_TYPE.jump
    elif var_name.endswith('log'):
        var_type = util.const.VAR_TYPE.log
    elif re.search('.*(?=_lag\d)', var_name) is not None:
        var_type = util.const.VAR_TYPE.lag
    else:
        var_type = util.const.VAR_TYPE.normal

    return var_type


def get_windows(time_scale_long, time_scale_short='3s'):
    td0 = pd.datetools.to_offset(time_scale_short).delta
    td1 = pd.datetools.to_offset(time_scale_long).delta
    windows = int(td1 / td0)
    return windows
