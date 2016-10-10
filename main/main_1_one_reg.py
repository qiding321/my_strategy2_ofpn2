# -*- coding: utf-8 -*-
"""
Created on 2016/10/8 10:46

@author: qiding
"""

import pickle as pkl

import data.data
import data.reg_data
import log.log
import my_path.path
import util.util
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

    # data_training = data.data.TrainingData(this_paras=my_para)
    # data_predicting = data.data.TestingData(this_paras=my_para)

    data_training = util.util.load_pkl(my_path.path.unit_test_data_path + 'data_training.pkl')
    data_predicting = util.util.load_pkl(my_path.path.unit_test_data_path + 'data_predicting.pkl')
    data_training.paras = my_para
    data_predicting.paras = my_para

    my_log.info('data end')
    assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)
    # util.util.dump_pkl(data_training, my_path.path.unit_test_data_path + 'data_training.pkl')
    # util.util.dump_pkl(data_predicting, my_path.path.unit_test_data_path + 'data_predicting.pkl')

    # ============================reg data=================
    reg_data_training, normalize_funcs = data_training.generate_reg_data()
    reg_data_testing, _ = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs, reg_data_training=reg_data_training)

    assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
    assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

    # ===========================reg and predict=====================
    reg_result = reg_data_training.fit()
    reg_data_testing.add_model(model=reg_data_training.model, paras_reg=reg_data_training.paras_reg)
    predict_result = reg_data_testing.predict()

    # ===========================record and analysis===================
    reg_data_testing.report_err_decomposition(output_path, file_name='error_decomposition.csv', predict_period=my_para.period_paras.begin_date_predict)
    reg_data_testing.report_daily_rsquared(output_path, file_name='daily_rsquared.csv')
    reg_data_testing.plot_daily_fitting(output_path + 'daily_fitting\\')
    reg_data_testing.plot_error_hist(output_path, file_name='error_hist.jpg')
    reg_data_testing.record_error_description(output_path, file_name='error_stats.csv')
    reg_data_training.plot_y_var_hist(output_path, file_name='y_var_hist_training.jpg')
    reg_data_testing.plot_y_var_hist(output_path, file_name='y_var_hist_predicting.jpg')

    data_training.report_description_stats(output_path, file_name='len_record_training.csv')
    data_predicting.report_description_stats(output_path, file_name='len_record_predicting.csv')


def unit_test():
    unit_test_path = my_path.path.unit_test_data_path
    unit_test_file1 = 'data_training.pkl'
    unit_test_file2 = 'data_predicting.pkl'
    with open(unit_test_path + unit_test_file1, 'rb') as f_in:
        data_training = pkl.load(f_in)
    with open(unit_test_path + unit_test_file2, 'rb') as f_in:
        data_predicting = pkl.load(f_in)

    reg_data_training, normalize_funcs = data_training.generate_reg_data()
    reg_data_testing, _ = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs, reg_data_training=reg_data_training)


if __name__ == '__main__':
    main()
    # unit_test()
