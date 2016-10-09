# -*- coding: utf-8 -*-
"""
Created on 2016/10/9 18:25

@author: qiding
"""

import my_path.path
import util.util


class TestPickle:
    def __init__(self):
        self.a = 1
        self.b = 2

    def func(self):
        def func_(x):
            return x

        return func_


tp1 = TestPickle()

util.util.dump_pkl(tp1, my_path.path.unit_test_data_path + 'pickle_test.pkl')
