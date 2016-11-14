# -*- coding: utf-8 -*-
"""
Created on 2016/11/10 16:37

@author: qiding
"""

import datetime
import time

import pandas as pd

import sql.sql

path_tmp = 'C:\\Users\\qiding\\Desktop\\reg_data_training_x.csv'


def main():
    t1 = time.clock()
    df = sql.sql.df_read_sql(table_name='reg_data_training_x')
    t2 = time.clock()
    df2 = pd.read_csv(path_tmp, date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'),
                      parse_dates=['time'])
    t3 = time.clock()
    df3 = sql.sql.df_read_sql(table_name='reg_data_training_x')
    t4 = time.clock()
    print(t2 - t1)
    print(t3 - t2)
    print(t4 - t3)


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
