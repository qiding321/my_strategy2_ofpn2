# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import socket

name = socket.gethostname()

if name == '2013-20151201LG':
    raw_data_source_path = 'I:\\stock_and_index_order_data_raw\\'
    data_source_path = 'I:\\OrderFlowPredictData_from_raw_data\\601818\\'
    market_making_result_root = 'E:\\StrategyResult\\MarketMaking\\'
    log_path_root = market_making_result_root + 'Log\\'
    unit_test_data_path = 'F:\\UnitTestData\\MarketMaking\\'
elif name == 'sas5':
    data_source_path = 'C:\\Users\\dqi\\Documents\\Data\\OrderFlowPredictData_from_raw_data\\601818\\'
    market_making_result_root = 'C:\\Users\\dqi\\Documents\\Output\\MarketMaking\\'
    log_path_root = market_making_result_root + 'Log\\'
    unit_test_data_path = 'C:\\Users\\dqi\\Documents\\Data\\UnitTestData\\MarketMaking\\'
else:
    raise NameError
