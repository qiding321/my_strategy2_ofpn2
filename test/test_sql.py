# -*- coding: utf-8 -*-
"""
Created on 2016/10/28 17:49

@author: qiding
"""

import pandas as pd
import pymysql

HOST = '192.168.1.183'
DB_NAME = 'qiding'
DB_USER = 'qd'
DB_PASS = '314159'
DB_PORT = 3306
DB_CHAR = 'utf8'

conn = pymysql.connect(host=HOST, db=DB_NAME, user=DB_USER, passwd=DB_PASS, port=DB_PORT, charset=DB_CHAR)

with conn.cursor() as cursor:
    # sql = 'select table_name from information_schema.tables where table_schema=\'csdb\' and table_type=\'base table\';;'
    sql = 'show tables;'
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)

df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=['aa', 'bb'], columns=['a', 'b', 'c'])
df.to_sql('df_', conn, 'mysql', if_exists='replace')

df2 = pd.read_sql('select * from df_;', conn)
