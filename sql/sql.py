# -*- coding: utf-8 -*-
"""
Created on 2016/10/28 17:21

@author: qiding
"""

import pandas as pd
import pymysql

import util.sql_config

HOST = util.sql_config.HOST
DB_NAME = util.sql_config.DB_NAME
DB_USER = util.sql_config.DB_USER
DB_PASS = util.sql_config.DB_PASS
DB_PORT = util.sql_config.DB_PORT
DB_CHAR = util.sql_config.DB_CHAR

conn = pymysql.connect(host=HOST, db=DB_NAME, user=DB_USER, passwd=DB_PASS, port=DB_PORT, charset=DB_CHAR)


def cursor(sql_code):
    with conn.cursor() as cursor:
        cursor.execute(sql_code)
        result = cursor.fetchall()
    return result


def df_to_sql(df, table_name, if_exists='replace'):
    df.to_sql(table_name, conn, 'mysql', if_exists=if_exists)


def df_read_sql(sql_code):
    df2 = pd.read_sql('select * from df_;', conn)
    return df2
