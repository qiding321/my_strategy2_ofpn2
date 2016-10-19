# -*- coding: utf-8 -*-
"""
Created on 2016/10/18 16:32

@author: qiding
"""

import smtplib
from email.header import Header
from email.mime.text import MIMEText

sender = 'dqi@mingshiim.com'
receivers = ['qiding321@126.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

# 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
message = MIMEText('content_test', 'plain', 'utf-8')
message['From'] = Header('dqi2', 'utf-8')
message['To'] = Header('dqi3', 'utf-8')

subject = 'subject_test'
message['Subject'] = Header(subject, 'utf-8')

mail_host = 'smtp.exmail.qq.com'
mail_user = 'dqi@mingshiim.com'
mail_pass = '****'

try:
    smtpObj = smtplib.SMTP()
    smtpObj.connect(mail_host, 25)  # 25 为 SMTP 端口号
    smtpObj.login(mail_user, mail_pass)
    smtpObj.sendmail(sender, receivers, message.as_string())
    print("邮件发送成功")
except smtplib.SMTPException as e:
    print("Error: 无法发送邮件")
    print(e)
