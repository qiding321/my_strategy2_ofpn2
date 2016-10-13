# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:47

@author: qiding
"""

import datetime

MARKET_OPEN_TIME = datetime.datetime(1900, 1, 1, 9, 30, 0)
MARKET_END_TIME = datetime.datetime(1900, 1, 1, 15, 0, 0)
MARKET_OPEN_TIME_NOON = datetime.datetime(1900, 1, 1, 13, 0, 0)
MARKET_CLOSE_TIME_NOON = datetime.datetime(1900, 1, 1, 11, 30, 0)

DAYS_ONE_YEAR = 252

MIN_TICK_SIZE = 100

# TIME_SCALE_LIST = ['1min', '5min', '10min', '15min']
TIME_SCALE_LIST = ['15s', '1min', '5min', '10min', '15min']


class FittingMethod:
    @property
    def OLS(self):
        return 'OLS'

    @property
    def DECTREEREG(self):
        return 'DecisionTreeRegression'

    @property
    def DECTREE(self):
        return 'DecisionTreeClassifier'

    @property
    def ADABOOST(self):
        return 'DecisionTreeAdaboost'

    @property
    def LOGIT(self):
        return 'Logit'


FITTING_METHOD = FittingMethod()


class VAR_TYPE:
    high_order = 'high_order'
    normal = 'normal'
    truncate = 'truncate'
    moving_average = 'moving_average'
    log = 'log'
    jump = 'jump'
    lag = 'lag'
    log_change = 'log_change'