# -*- coding: utf-8 -*-
"""
Created on 2017/1/13 9:25

@version: python3.5
@author: qiding
"""

import os

import data.data
import data.reg_data
import log.log
import my_path.path
import util.const
import util.util
from paras.paras import Paras


def main():
    # ==========================parameters and path======================
    my_para = Paras()
    output_path = 'F:\\MMRegPara\\' + my_para.get_title() + '\\'

    # =========================log================================
    my_log = log.log.log_order_flow_predict
    my_log.add_path(log_path2=output_path + 'log.log')
    my_log.info('paras:\n%s' % my_para)
    my_log.info('output path:\n{}'.format(output_path))

    # ============================loading data from csv====================
    my_log.info('data begin')
    data_base = data.data.RolldingData(this_paras=my_para)
    # util.util.dump_pkl(data_base, my_path.path.unit_test_data_path + 'data_base.pkl')
    # data_base = util.util.load_pkl(my_path.path.unit_test_data_path + 'data_base.pkl')
    # data_base.paras = my_para
    my_log.info('data end')
    for data_dict_ in data_base.generate_rolling_data():
        data_training, data_predicting, data_demean = [
            data_dict_[col] for col in ['data_training', 'data_predicting', 'data_out_of_sample_demean']
            ]
        assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)
        output_path_this_month = output_path + data_predicting.date_begin + '\\'
        if not os.path.exists(output_path_this_month):
            os.makedirs(output_path_this_month)

        # ============================reg data=================
        reg_data_training, normalize_funcs = data_training.generate_reg_data()
        reg_data_testing, _ = data_predicting.generate_reg_data(normalize_funcs=normalize_funcs,
                                                                reg_data_training=reg_data_training)

        assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
        assert isinstance(reg_data_testing, data.reg_data.RegDataTest)
        # corr matrix
        reg_data_training.report_corr_matrix(output_path_this_month, file_name='corr_training.csv')
        reg_data_testing.report_corr_matrix(output_path_this_month, file_name='corr_testing.csv')
        reg_data_training.report_variance(output_path_this_month, file_name='variance_training.csv')
        reg_data_testing.report_variance(output_path_this_month, file_name='variance_testing.csv')

        # ===========================reg and predict=====================
        reg_result = reg_data_training.fit()
        reg_data_testing.add_model(model=reg_data_training.model, paras_reg=reg_data_training.paras_reg)
        predict_result = reg_data_testing.predict()

        # ===========================record and analysis===================

        reg_data_training.save_reg_paras_to_pickle(output_path_this_month + 'reg_paras.pkl')
        reg_data_training.save_reg_paras_to_csv(output_path_this_month + 'reg_paras.csv')

        # in sample summary
        reg_data_training.report_summary(output_path_this_month, file_name='reg_summary.txt')
        # daily
        reg_data_testing.report_daily_rsquared(
            output_path_this_month,
            file_name=('daily_rsquared.csv', 'daily_rsquared.jpg')
        )
        # reg_data_testing.plot_daily_fitting(output_path_this_month + 'daily_fitting\\')
        # var analysis
        bars_, max_bar_accuracy_in_sample, _, _ = reg_data_training.report_risk_analysis(
            output_path_this_month + 'var_analysis\\', 'in_sample10', percent_num=10
        )
        _, max_bar_accuracy_oos, _, _ = reg_data_testing.report_risk_analysis(
            output_path_this_month + 'var_analysis\\', 'out_of_sample10', bars=bars_
        )
        bars_, max_bar_accuracy_in_sample, _, _ = reg_data_training.report_risk_analysis(
            output_path_this_month + 'var_analysis\\', 'in_sample40', percent_num=40
        )
        _, max_bar_accuracy_oos, _, _ = reg_data_testing.report_risk_analysis(
            output_path_this_month + 'var_analysis\\', 'out_of_sample40', bars=bars_
        )
        _, max_bar_accuracy_oos, _, _ = reg_data_testing.report_risk_analysis(
            output_path_this_month + 'var_analysis\\', 'out_of_sample_cutoff', bars=[0, .2, .4, .6, .8, 1]
        )
        # error
        if my_para.method_paras.method not in [util.const.FITTING_METHOD.GARCH, util.const.FITTING_METHOD.DECTREE]:
            reg_data_testing.report_err_decomposition(
                output_path_this_month, file_name='error_decomposition.csv',
                predict_period=my_para.period_paras.begin_date_predict
            )
        reg_data_testing.plot_error_hist(output_path_this_month, file_name='error_hist')
        reg_data_testing.record_error_description(output_path_this_month, file_name='error_stats.csv')
        # hist
        reg_data_training.plot_y_var_hist(output_path_this_month, file_name='y_var_hist_training')
        reg_data_training.plot_x_var_hist(output_path_this_month + 'x_var_hist_training\\')
        reg_data_testing.plot_y_var_hist(output_path_this_month, file_name='y_var_hist_testing')
        reg_data_training.predict_y_hist(output_path_this_month, file_name='y_predict_hist_training')
        reg_data_testing.predict_y_hist(output_path_this_month, file_name='y_predict_hist_testing')

        # data length
        data_training.report_description_stats(output_path_this_month, file_name='len_record_training.csv')
        data_predicting.report_description_stats(output_path_this_month, file_name='len_record_predicting.csv')
        # resume data if it is taken log
        reg_data_testing.report_resume_if_logged(output_path_this_month + 'resumed_data_record\\')

if __name__ == '__main__':
    main()
    # unit_test()
