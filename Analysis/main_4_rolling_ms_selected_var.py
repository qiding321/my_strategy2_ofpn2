import os

import pandas as pd

import log.log
import util.const


def get_path_list(path_root):
    path_list = []
    for folder1 in os.listdir(path_root):
        if not os.path.isdir(path_root + folder1):
            continue
        path_list.append(path_root+folder1+'\\')
    return path_list


def main():
    path_root = r'E:\StrategyResult\MarketMaking'+'\\'+r'2016-12-04-12-23-00buy_jump_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_'+'\\'

    reg_type = util.const.FITTING_METHOD.LOGIT
    # reg_type = util.const.FITTING_METHOD.OLS

    my_log = log.log.log_order_flow_predict

    date_path_list = get_path_list(path_root)
    print('\n'.join(date_path_list))


    record_list = {}

    for date_path in date_path_list:
        this_date = date_path.split('\\')[-2]
        if reg_type == util.const.FITTING_METHOD.LOGIT:
            # file_path = date_path + 'var_analysis\\out_of_samplepercentile.csv'
            file_path = date_path + 'var_analysis\\out_of_sample10percentile.csv'
            data_ = pd.read_csv(file_path)
            max_value = data_['accuracy'].iloc[-1]
            # max_value = data_['target_num'].sum()/data_['all_num'].sum()

        else:
            file_path = date_path + 'daily_rsquared.csv'
            data_ = pd.read_csv(file_path)
            msr = data_['msr'].sum()
            mse = data_['mse'].sum()
            max_value = 1 - msr / mse
        record_list[this_date] = max_value
    record_df = pd.Series(record_list)
    record_df.to_csv(path_root+'accuracy_record.csv')
    # record_df.to_csv(path_root+'accuracy_record_actual.csv')


if __name__ == '__main__':
    main()
