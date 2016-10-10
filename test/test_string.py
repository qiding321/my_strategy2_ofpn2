# -*- coding: utf-8 -*-
"""
Created on 2016/10/10 14:15

@author: qiding
"""

utf = r'123abc啦啦啦'
b = b'123abc\xe5\x95\xa6\xe5\x95\xa6\xe5\x95\xa6'

utf_encode_gbk = utf.encode('gbk')
utf_encode_utf = utf.encode('utf8')

b_decode_utf = utf_encode_utf.decode('utf8')
b_decode_gbk = utf_encode_gbk.decode('gbk')
