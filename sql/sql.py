# -*- coding: utf-8 -*-
"""
Created on 2016/10/28 17:21

@author: qiding
"""

import pandas as pd
import pymysql

import log.log
import util.sql_config

my_log = log.log.log_order_flow_predict

HOST = util.sql_config.HOST
DB_NAME = util.sql_config.DB_NAME
DB_USER = util.sql_config.DB_USER
DB_PASS = util.sql_config.DB_PASS
DB_PORT = util.sql_config.DB_PORT
DB_CHAR = util.sql_config.DB_CHAR

conn = pymysql.connect(host=HOST, db=DB_NAME, user=DB_USER, passwd=DB_PASS, port=DB_PORT, charset=DB_CHAR)
my_log.info('sql connected')


def cursor(sql_code):
    with conn.cursor() as cursor:
        cursor.execute(sql_code)
        result = cursor.fetchall()
    return result


def df_to_sql(df, table_name, if_exists='replace'):
    my_log.info('df_to_sql: {}'.format(table_name))
    if isinstance(df, pd.DataFrame):
        pass
    else:
        df = pd.DataFrame(df)
    df.to_sql(table_name, conn, 'mysql', if_exists=if_exists)


def df_read_sql(sql_code=None, table_name=None):
    my_log.info('df_read_sql: {}'.format(table_name))

    if sql_code is not None:
        df2 = pd.read_sql(sql_code, conn)
    elif table_name is not None:
        df2 = pd.read_sql('select * from {}'.format(table_name), conn)
    else:
        raise ValueError

    df2 = df2.set_index('time')

    return df2
