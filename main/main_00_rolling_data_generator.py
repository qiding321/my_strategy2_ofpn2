# -*- coding: utf-8 -*-
"""
Created on 2017/1/14 10:47

@version: python3.5
@author: qiding
"""

import datetime
import os
import pickle

import pandas as pd

import paras.paras
import paras.resample_dicts as resample_dicts
import util.const as const
import util.util as util
from log.log import log_error
from log.log import log_order_flow_predict


def main():
    output_folder = 'I:\\stock_tick_data_with_reg_result\\601818\\'

    stk_input_folder = 'I:\\stock_and_index_order_data_raw\\Tick\\SH\\'
    trans_input_folder = 'I:\\stock_and_index_order_data_raw\\Transaction\\SH\\'
    index_input_folder = 'I:\\stock_and_index_order_data_raw\\Tick\\SH\\'
    index_future_input_folder = 'F:\\IF_complete_data\\'
    # stk_input_folder = 'I:\\IntradayDataOwT\\SH\\'
    # index_input_folder = 'F:\\IntradayIndex\\Tick\\SH\\'

    reg_model_path = r'F:\MMRegPara' + '\\'

    start_time = '20140901'

    input_output_file_mapping_list = generate_file_mapping_list(stk_input_folder, index_input_folder, trans_input_folder, index_future_input_folder,
                                                                output_folder, start_time)  # [(index_in, stk_in, transaction_in, index_future_in, file_out)]
    log_order_flow_predict.info('file_mapping_list: %d' % len(input_output_file_mapping_list))

    for index_in, stk_in, transaction_in, index_future_in, file_out in input_output_file_mapping_list:
        log_order_flow_predict.info(file_out + ' begin')

        index_data_50 = pd.read_csv(index_in[0], parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'), encoding='gbk')[
            ['date', 'time', 'price', 'volume', 'accvolume']].drop_duplicates('time', keep='last').set_index('time')
        index_data_300 = pd.read_csv(index_in[1], parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'), encoding='gbk')[
            ['date', 'time', 'price', 'volume', 'accvolume']].drop_duplicates('time', keep='last').set_index('time')
        stk_data = pd.read_csv(stk_in, parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'),
                               encoding='gbk').drop_duplicates('time', keep='last').set_index('time')
        transaction_data = pd.read_csv(transaction_in, parse_dates=['time'], date_parser=lambda x: datetime.datetime.strptime(str(x), '%H%M%S%f'),
                                       encoding='gbk').set_index('time')
        index_future_data = _get_index_future_data(index_future_in)

        idx = stk_data.index

        index_data_50_resample = clean_index_data(index_data_50, idx, name='sh50')
        index_data_300_resample = clean_index_data(index_data_300, idx, name='hs300')
        index_fut_data_resample = clean_fut_data(index_future_data, idx, name='index_future_300')
        stk_tran_data = merge_stk_trans_data(stk_data, transaction_data)

        data_merged = pd.DataFrame(pd.concat([
            stk_tran_data,
            index_data_50_resample,
            index_data_300_resample,
            index_fut_data_resample,
        ], axis=1))

        today_str = str(int(round(data_merged['date'].iloc[0],2)))
        this_month_1st_day_str = today_str[0:6] + '01'
        with open(reg_model_path + '2017-01-13-17-53-23buy_mean_selected_manually__normalize_F_divide_std_F_OLS_truncate_period30_std4_'+'\\'+this_month_1st_day_str+'\\reg_paras.pkl', 'rb') as f_model:
            buy_mean_model = pickle.load(f_model)
        with open(reg_model_path + '2017-01-13-17-54-08sell_mean_selected_manually__normalize_F_divide_std_F_OLS_truncate_period30_std4_'+'\\'+this_month_1st_day_str+'\\reg_paras.pkl', 'rb') as f_model:
            sell_mean_model = pickle.load(f_model)
        with open(reg_model_path + '2017-01-13-17-13-20buy_jump_selected_rolling_cutoffand3month_strict__normalize_F_divide_std_F_Logit_truncate_period30_std4_'+'\\'+this_month_1st_day_str+'\\reg_paras.pkl', 'rb') as f_model:
            buy_jump_model = pickle.load(f_model)
        with open(reg_model_path + '2017-01-13-17-50-14sell_jump_qd_manually_selected__normalize_F_divide_std_F_Logit_truncate_period30_std4_'+'\\'+this_month_1st_day_str+'\\reg_paras.pkl', 'rb') as f_model:
            sell_jump_model = pickle.load(f_model)

        para_buy_mean = paras.paras.XvarsParaBuyMeanSelectedManually()
        para_sell_mean = paras.paras.XvarsParaSellMeanSelectedManually()
        para_buy_jump = paras.paras.XvarsParaBuyJumpSelectedCutoff3MonthStrict()
        para_sell_jump = paras.paras.XvarsParaSellJumpQDManuallySelected()

        buy_mean_predict = generate_buy_mean_predict(data_merged, buy_mean_model, para_buy_mean)
        # sell_mean_predict = generate_sell_mean_predict(data_merged, sell_mean_model)
        # buy_jump_predict = generate_buy_jump_predict(data_merged, buy_jump_model)
        # sell_jump_predict = generate_sell_jump_predict(data_merged, sell_jump_model)

        # data_final = pd.DataFrame(pd.concat([
        #     buy_mean_predict,
        #     sell_mean_predict,
        #     buy_jump_predict,
        #     sell_jump_predict,
        # ], axis=1))

        log_order_flow_predict.info(file_out + ' end')
        #
        # data_final.to_csv(file_out)


def generate_buy_mean_predict(data_merged, buy_mean_model, para_buy_mean):
    pass


def clean_index_data_rolling(index_data, idx, name='', window='1min'):
    for key_list_tmp, idx_lead in idx_iter(idx_lag_iter=index_data.index.__iter__, idx_lead_iter=idx.__iter__):
        data_lag = index_data.loc[key_list_tmp, :]
        volume = data_lag['volume'].sum()
        ret = data_lag['price'].iloc[-1] / data_lag['price'].iloc[0] - 1

    pass

def idx_iter(idx_lead_iter, idx_lag_iter, name='', window='1min'):
    if window == '1min':
        time_delta = datetime.timedelta(seconds=60)
    else:
        raise ValueError

    idx_lead = idx_lead_iter.__next__()
    key_lag, row =  idx_lag_iter.__next__()

    while True:
        key_list_tmp = []
        while True:
            if key_lag >= idx_lead:
                break
            try:
                key_lag, row = idx_lag_iter.__next__()
            except StopIteration:
                break
            key_list_tmp.append(key_lag)

        # todo, do something with key list tmp
        yield key_list_tmp, idx_lead

        try:
            idx_lead = idx_lead_iter.__next__()
        except StopIteration:
            raise StopIteration
        remove_list = []
        for key_ in key_list_tmp:
            if key_ < idx_lead - time_delta:
                remove_list.append(key_)
            else:
                break
        for to_remove in remove_list:
            key_list_tmp.remove(to_remove)


def merge_stk_trans_data(stk_data, transaction_data):
    iter_tick = stk_data.iterrows()
    iter_trans = transaction_data.iterrows()

    key_tick, row_tick = iter_tick.__next__()
    acc_vol_order = row_tick['accvolume']

    key_trans, row_trans = iter_trans.__next__()
    acc_vol_trans = row_trans['trade_volume']
    key_tick, row_tick = iter_tick.__next__()
    acc_vol_order = row_tick['accvolume']

    data_list = []
    key_list = []

    while True:
        buy_vol = 0
        sell_vol = 0
        while True:
            if acc_vol_trans > acc_vol_order:
                assert acc_vol_trans - acc_vol_order == row_trans['trade_volume']
                break
            else:
                pass
            buy_vol += row_trans['trade_volume'] if row_trans['bs_flag'] == ord('B') else 0
            sell_vol += row_trans['trade_volume'] if row_trans['bs_flag'] == ord('S') else 0
            try:
                key_trans, row_trans = iter_trans.__next__()
                acc_vol_trans += row_trans['trade_volume']
            except StopIteration:
                row_trans = pd.Series()
                break
        row_tick['buy_vol'] = buy_vol
        row_tick['sell_vol'] = sell_vol
        data_list.append(row_tick)
        key_list.append(key_tick)

        try:
            key_tick, row_tick = iter_tick.__next__()
            acc_vol_order = row_tick['accvolume']
        except StopIteration:
            # row_tick = pd.Series()
            break
    stk_tran_data = pd.DataFrame(data_list)

    return stk_tran_data


def _get_index_future_data(index_future_path):
    data = pd.read_csv(index_future_path, encoding='gbk', date_parser=lambda x: datetime.datetime.strptime(str(x)[11:], '%H:%M:%S.%f'),
                       parse_dates=['时间'])  # todo
    data_new_name = data.rename(columns={'时间': 'time', '最新': 'price', '成交量': 'volume'})
    data_new_name['accvolume'] = data_new_name['volume'].cumsum()

    data_to_ret = data_new_name[['time', 'price', 'volume', 'accvolume']].drop_duplicates('time', keep='last').set_index('time')
    return data_to_ret


def _date_time_merge(time_, date_):
    year = int(str(date_)[0:4])
    month = int(str(date_)[4:6])
    day = int(str(date_)[6:8])
    hour = time_.hour
    minute = time_.minute
    second = time_.second
    time = datetime.datetime(year, month, day, hour, minute, second)
    return time


def _get_index_future_path(path_root, date):
    path0 = path_root + 'sf2014\\' + date + '\\'
    if os.path.exists(path0):
        pass
    else:
        path0 = path_root + 'sf2013\\' + date + '\\'
    if os.path.exists(path0):
        pass
    else:
        return None
    file_name_list = os.listdir(path0)
    file_name_list_if = sorted([f_n_ for f_n_ in file_name_list if f_n_[0:2] == 'IF' and f_n_[-3:] == 'csv'])
    file_name = file_name_list_if[0]
    return path0 + file_name


def generate_file_mapping_list(stk_input_folder, index_input_folder, trans_input_folder, index_future_input_folder, output_folder, start_time=None):
    dates1 = os.listdir(index_input_folder)
    dates2 = os.listdir(stk_input_folder)
    dates = [d_ for d_ in dates1 if d_ in dates2 and (d_ >= start_time if start_time is not None else True)]
    mapping_list = []
    for date in dates:
        index_in1 = index_input_folder + date + '\\' + '999987.csv'
        index_in2 = index_input_folder + date + '\\' + '000300.csv'
        stk_in = stk_input_folder + date + '\\' + '601818.csv'
        transaction_in = trans_input_folder + date + '\\' + '601818.csv'
        file_out = output_folder + date + '.csv'
        index_future_in = _get_index_future_path(index_future_input_folder, date)
        if index_future_in is None:
            log_error.info('file not exist: %s' % index_future_input_folder + date)
        elif all(os.path.exists(path_) for path_ in [index_in1, index_in2, stk_in, transaction_in, index_future_in]):
            mapping_list.append(((index_in1, index_in2), stk_in, transaction_in, index_future_in, file_out))
        else:
            [
                log_error.info('file not exist: %s' % file_path_)
                for file_path_ in [index_in1, index_in2, stk_in, transaction_in, index_future_in]
                if not os.path.exists(file_path_)
                ]

    return mapping_list


def clean_index_data(index_data, idx, name):  # todo, check
    index_data_filtered = filter_time(index_data)

    fun_dict = {
        'date'     : 'last',
        'price'    : 'last',
        'volume'   : 'sum',
        'accvolume': 'last'
    }
    rename_dict = {
        'date'     : 'date_index' + '_' + name,
        'price'    : 'price_index' + '_' + name,
        'volume'   : 'volume_index' + '_' + name,
        'accvolume': 'accvolume_index' + '_' + name
    }

    resample_data = util.resample_to_index(data_source=index_data_filtered, idx_destn=idx, funcs=fun_dict).rename(columns=rename_dict)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_index(index_data, freq='3s', name=''):
    index_data_filtered = filter_time(index_data)

    fun_dict = {
        'date'     : 'last',
        'price'    : 'last',
        'volume'   : 'sum',
        'accvolume': 'last'
    }
    rename_dict = {
        'date'     : 'date_index' + '_' + name,
        'price'    : 'price_index' + '_' + name,
        'volume'   : 'volume_index' + '_' + name,
        'accvolume': 'accvolume_index' + '_' + name
    }

    resample_data = index_data_filtered.resample(freq, label='right', closed='left').apply(fun_dict).rename(columns=rename_dict)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_fut_data(index_data, idx, name):
    index_data_filtered = filter_time(index_data)

    fun_dict = {
        'price'    : 'last',
        'volume'   : 'sum',
        'accvolume': 'last'
    }
    rename_dict = {
        'price'    : 'price_index' + '_' + name,
        'volume'   : 'volume_index' + '_' + name,
        'accvolume': 'accvolume_index' + '_' + name
    }

    resample_data = util.resample_to_index(data_source=index_data_filtered, idx_destn=idx, funcs=fun_dict).rename(columns=rename_dict)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_index_future(index_data, freq='3s', name=''):
    index_data_filtered = filter_time(index_data)

    fun_dict = {
        'price'    : 'last',
        'volume'   : 'sum',
        'accvolume': 'last'
    }
    rename_dict = {
        'price'    : 'price_index' + '_' + name,
        'volume'   : 'volume_index' + '_' + name,
        'accvolume': 'accvolume_index' + '_' + name
    }

    resample_data = index_data_filtered.resample(freq, label='right', closed='left').apply(fun_dict).rename(columns=rename_dict)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_stk(stk_data, freq='3s'):
    func_dict = resample_dicts.stk_func_dict

    cols = list(set(stk_data.columns).intersection(set(func_dict.keys())))
    func_dict2 = dict((x_, func_dict[x_]) for x_ in cols)

    resample_data = stk_data.resample(freq, label='right', closed='left')[cols].agg(func_dict2)
    resample_data2 = filter_time(resample_data)

    return resample_data2


def clean_data_transaction(transaction_data, freq='3s'):
    transaction_data['amount'] = transaction_data['trade_price'] * transaction_data['trade_volume']

    transaction_data_buy = transaction_data[transaction_data['bs_flag'] == ord('B')]
    transaction_data_sell = transaction_data[transaction_data['bs_flag'] == ord('S')]

    columns = [
        'newprice',
        'totalamount', 'totalvolume', 'totaltransaction',
        'buytrans', 'selltrans',
        'buyvolume', 'sellvolume',
        'buyamount', 'sellamount'
    ]

    def resample_sum(s_):
        return s_.resample(freq, label='right', closed='left').sum()

    def resample_last(s_):
        return s_.resample(freq, label='right', closed='left').apply('last')

    def resample_count(s_):
        return s_.resample(freq, label='right', closed='left').count()

    newprice = resample_last(transaction_data['trade_price'])
    totalamount = resample_sum(transaction_data['amount'])
    totalvolume = resample_sum(transaction_data['trade_volume'])
    totaltransaction = resample_count(transaction_data['trade_volume'])

    buyamount = resample_sum(transaction_data_buy['amount'])
    buyvolume = resample_sum(transaction_data_buy['trade_volume'])
    buytrans = resample_count(transaction_data_buy['trade_volume'])

    sellamount = resample_sum(transaction_data_sell['amount'])
    sellvolume = resample_sum(transaction_data_sell['trade_volume'])
    selltrans = resample_count(transaction_data_sell['trade_volume'])

    resample_data = pd.DataFrame([
        newprice,
        totalamount, totalvolume, totaltransaction,
        buytrans, selltrans,
        buyvolume, sellvolume,
        buyamount, sellamount
    ], index=columns).T

    resample_data2 = filter_time(resample_data)

    return resample_data2


def filter_time(data):
    data_new = data.select(lambda x: const.MARKET_OPEN_TIME <= x <= const.MARKET_CLOSE_TIME_NOON or const.MARKET_OPEN_TIME_NOON <= x <= const.MARKET_END_TIME)
    return data_new


if __name__ == '__main__':
    main()
