# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import data.data
import data.reg_data
import log.log
import my_path.path
from paras.paras import Paras


def main():
    # ==========================parameters and path======================
    my_para = Paras()
    output_path = my_path.path.market_making_result_root + my_para.get_title() + '\\'

    # =========================log================================
    my_log = log.log.log_order_flow_predict
    my_log.add_path(log_path2=output_path + 'log.log')
    my_log.info('paras:\n%s' % my_para)
    my_log.info('output path:\n{}'.format(output_path))

    # ============================loading data from csv====================
    my_log.info('data begin')
    data_training = data.data.TrainingData(this_paras=my_para)
    data_predicting = data.data.TestingData(this_paras=my_para)
    my_log.info('data end')
    assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)

    # ============================reg data=================
    reg_data_training, normalize_funcs = data_training.generate_reg_data()
    reg_data_testing, _ = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs)

    assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
    assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

    # ===========================reg and predict=====================
    reg_result = reg_data_training.fit()
    reg_data_testing.add_model(model=reg_data_training.model, paras_reg=reg_data_training.paras_reg)
    predict_result = reg_data_testing.predict()

    # ===========================record and analysis===================
    # # output_file
    # output_file = output_path + 'r_squared_record.csv'
    # with open(output_file, 'w') as f_out:
    #     f_out.write('time_period_in_sample,time_period_out_of_sample,rsquared_in_sample,rsquared_out_of_sample\n')

    # r_sq_in_sample = util.util.cal_r_squared(y_raw=reg_data_training.y_vars.values.T[0], y_predict=reg_data_training.y_predict_insample,
    #                                          y_training=reg_data_training.y_vars.values.T[0])
    # r_sq_out_of_sample = util.util.cal_r_squared(y_raw=reg_data_testing.y_vars.values.T[0], y_predict=reg_data_testing.predict_y.T,
    #                                              y_training=reg_data_training.y_vars.values.T[0])
    # with open(output_file, 'a') as f_out:
    #     to_record = '{time_period_in_sample},{time_period_out_of_sample},{rsquared_in_sample},{rsquared_out_of_sample}\n'.format(
    #         time_period_in_sample=in_sample_period, time_period_out_of_sample=out_of_sample_period, rsquared_in_sample=r_sq_in_sample,
    #         rsquared_out_of_sample=r_sq_out_of_sample
    #     )
    #     f_out.write(to_record)
    # time_period_name = out_of_sample_period
    # err_testing = reg_data_testing.get_err()
    # reg_data_testing.report_accuracy(output_path=output_path, name=time_period_name)
    # reg_data_testing.report_err(output_path, err_testing, name=time_period_name)
    # reg_data_testing.report_monthly(output_path, name_time_period=time_period_name, normalize_funcs=normalize_funcs,
    #                                 normalize_funcs_training=normalize_funcs_useless)
    #
    # data_training.report_description_stats(output_path, name_time_period=time_period_name, file_name='len_record_training.csv')
    # data_predicting.report_description_stats(output_path, name_time_period=time_period_name, file_name='len_record_predicting.csv')
