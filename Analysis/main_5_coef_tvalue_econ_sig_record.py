import os

import numpy as np
import pandas as pd

import log.log
import util.const
from Analysis.func_1_model_selection_modules import OneRegResult


def get_path_list(path_root):
    path_list = []
    for folder1 in os.listdir(path_root):
        if not os.path.isdir(path_root + folder1):
            continue
        path_list.append(path_root+folder1+'\\')
    return path_list


def main():
    path_root = r'E:\StrategyResult\MarketMaking'+'\\'+r'2016-12-02-15-15-09sell_jump_selected_rolling_cutoffand3month__normalize_F_divide_std_F_Logit_truncate_period30_std4_'+'\\'

    reg_type = util.const.FITTING_METHOD.LOGIT
    # reg_type = util.const.FITTING_METHOD.OLS
    para_type = 'marginal_effect'

    my_log = log.log.log_order_flow_predict

    date_path_list = get_path_list(path_root)

    output_path = path_root + ('coef_record.csv' if para_type != 'marginal_effect' else 'marginal_effect_record.csv')
    with open(output_path, 'w') as f_out:
        f_out.write('')

    for date_path in date_path_list:
        this_date = date_path.split('\\')[-2]

        this_reg = OneRegResult(result_path=date_path, reg_type=reg_type, para_type=para_type)
        try:
            this_reg.update_vars_from_path()
            p_value_this_reg = this_reg.p_value
            coef_this_reg = this_reg.coef
            var_ = pd.read_csv(date_path+'variance_training.csv', index_col=[0], header=None)[1]
            var__ = var_.rename(dict(map(lambda x: (x, x+'_x'), var_.index)))
            econ_sig_this_reg = dict(zip(coef_this_reg.keys(), [np.sqrt(var__[k])*float(v) for k, v in coef_this_reg.items()]))

            data_this_date_pd = pd.DataFrame([p_value_this_reg, coef_this_reg, econ_sig_this_reg, var__], index=['p_value', 'coef', 'econ_sig', 'variance']).T

            with open(output_path, 'a') as f_out:
                f_out.write(this_date+'\n')
                s = data_this_date_pd.to_csv()
                f_out.write(s+'\n')

        except FileNotFoundError:
            my_log.error(date_path)
            continue

        # record_list[this_date] = max_value
    # record_df = pd.Series(record_list)
    # record_df.to_csv(path_root+'accuracy_record.csv')
    # record_df.to_csv(path_root+'accuracy_record_actual.csv')


if __name__ == '__main__':
    main()
