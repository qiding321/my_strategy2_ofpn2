# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""


class RegData:
    def __init__(self, x_vars, y_vars, paras_config):
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.paras_config = paras_config


class RegDataTraining(RegData):
    pass


class RegDataTest(RegData):
    pass
