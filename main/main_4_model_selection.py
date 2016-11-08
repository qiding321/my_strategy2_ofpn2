# -*- coding: utf-8 -*-
"""
Created on 2016/11/1 14:48

@author: qiding
"""

import data.data
import data.model_selection
import data.reg_data
import log.log
import my_path.path
import paras.paras
import sql.sql
import util.const
import util.util


def main():
    # ==========================parameters and path======================
    my_para = paras.paras.ParasModelSelection()
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
    # util.util.dump_pkl(data_training, my_path.path.unit_test_data_path + 'data_training.pkl')
    # util.util.dump_pkl(data_predicting, my_path.path.unit_test_data_path + 'data_predicting.pkl')

    data_training = util.util.load_pkl(my_path.path.unit_test_data_path + 'data_training.pkl')
    data_predicting = util.util.load_pkl(my_path.path.unit_test_data_path + 'data_predicting.pkl')
    data_training.paras = my_para
    data_predicting.paras = my_para

    my_log.info('data end')
    assert isinstance(data_training, data.data.TrainingData) and isinstance(data_predicting, data.data.TestingData)
    # my_log.info('data to sql begin')
    # sql.sql.df_to_sql(df=data_training.data_df, table_name='training_data_df_temp')
    # sql.sql.df_to_sql(df=data_predicting.data_df, table_name='testing_data_df_temp')
    # my_log.info('data to sql end')

    # ============================reg data=================
    # x_vars_training = sql.sql.df_read_sql(table_name='reg_data_training_x')
    # y_vars_training = sql.sql.df_read_sql(table_name='reg_data_training_y')
    # x_vars_testing = sql.sql.df_read_sql(table_name='reg_data_testing_x')
    # y_vars_testing = sql.sql.df_read_sql(table_name='reg_data_testing_y')
    # reg_data_training, normalize_funcs = data_training.generate_reg_data(x_series=x_vars_training,
    #                                                                      y_series=y_vars_training)
    # reg_data_testing, _ = data_predicting.generate_reg_data(
    #     normalize_funcs=normalize_funcs, reg_data_training=reg_data_training,
    #     x_series=x_vars_testing, y_series=y_vars_testing
    # )

    reg_data_training, normalize_funcs = data_training.generate_reg_data()
    reg_data_testing, _ = data_predicting.generate_reg_data(
        normalize_funcs=normalize_funcs, reg_data_training=reg_data_training,
    )
    sql.sql.df_to_sql(df=reg_data_training.y_predict, table_name='data_training_predicted_y')
    sql.sql.df_to_sql(df=reg_data_testing.y_predict, table_name='data_testing_predicted_y')
    sql.sql.df_to_sql(df=reg_data_training.x_vars, table_name='data_training_x')
    sql.sql.df_to_sql(df=reg_data_testing.x_vars, table_name='data_testing_x')
    my_log.info('df to sql done')

    assert isinstance(reg_data_training, data.reg_data.RegDataTraining)
    assert isinstance(reg_data_testing, data.reg_data.RegDataTest)

    # ===========================reg and predict=====================

    model_selection = data.model_selection.ModelSelection(reg_data_training, reg_data_testing)
    for model_left_len in model_selection.iter_model_len():
        max_accuracy_list = []
        vars_del_list = []
        for reg_data_training, reg_data_testing, vars_del in model_selection.iter_model_config():
            output_path2 = model_selection.get_name(output_path)
            reg_result = reg_data_training.fit()
            reg_data_testing.add_model(model=reg_data_training.model, paras_reg=reg_data_training.paras_reg)
            predict_result = reg_data_testing.predict()

            # ===========================record and analysis===================
            # in sample summary
            reg_data_training.report_summary(output_path2, file_name='reg_summary.txt')
            # daily
            # reg_data_testing.report_daily_rsquared(output_path,
            #                                        file_name=('daily_rsquared.csv', 'daily_rsquared.jpg'))
            # reg_data_testing.plot_daily_fitting(output_path + 'daily_fitting\\')

            # var analysis
            # bars_, max_bar_accuracy_in_sample, _, _ \
            #     = reg_data_training.report_risk_analysis(output_path2 + 'var_analysis\\', 'in_sample')
            # _, max_bar_accuracy_oos, max_bar_hit, max_bar_len \
            #     = reg_data_testing.report_risk_analysis(output_path2 + 'var_analysis\\', 'out_of_sample', bars=bars_)

            r_squared_oos = reg_data_testing.get_r_squared()
            max_bar_accuracy_oos, max_bar_hit, max_bar_len = r_squared_oos, 0, 0

            max_accuracy_list.append(r_squared_oos)
            vars_del_list.append(vars_del)
            model_selection.record_vars(path_=output_path2)
            model_selection.record_result(output_path, max_bar_accuracy_oos,
                                          max_bar_hit, max_bar_len)
            # error
            # if my_para.method_paras.method not in [util.const.FITTING_METHOD.GARCH, util.const.FITTING_METHOD.DECTREE]:
            #     reg_data_testing.report_err_decomposition(output_path, file_name='error_decomposition.csv',
            #                                               predict_period=my_para.period_paras.begin_date_predict)
            # reg_data_testing.plot_error_hist(output_path, file_name='error_hist')
            # reg_data_testing.record_error_description(output_path, file_name='error_stats.csv')
            # hist
            # reg_data_training.plot_y_var_hist(output_path, file_name='y_var_hist_training')
            # reg_data_training.plot_x_var_hist(output_path + 'x_var_hist_training\\')
            # reg_data_testing.plot_y_var_hist(output_path, file_name='y_var_hist_testing')
            # reg_data_training.predict_y_hist(output_path, file_name='y_predict_hist_training')
            # reg_data_testing.predict_y_hist(output_path, file_name='y_predict_hist_testing')
            # data length
            # data_training.report_description_stats(output_path, file_name='len_record_training.csv')
            # data_predicting.report_description_stats(output_path, file_name='len_record_predicting.csv')
            # resume data if it is taken log
            # reg_data_testing.report_resume_if_logged(output_path + 'resumed_data_record\\')
        max_idx = max_accuracy_list.index(max(max_accuracy_list))
        vars_del_max = vars_del_list[max_idx]
        model_selection.del_var(vars_del_max)


if __name__ == '__main__':
    main()
