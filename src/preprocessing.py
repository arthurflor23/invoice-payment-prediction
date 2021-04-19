from tqdm import tqdm
from functools import partial

import multiprocessing
import pandas as pd
import numpy as np
import argparse
import os


def step0_extract(filename):
    df = pd.read_csv(filename, low_memory=False)
    print('Total size:\t\t\t\t\t', len(df))

    df = df[df['TransactionType'] == 'FI-InvoicedDocument']
    print('Total size (transaction type filter):\t\t', len(df))

    df = df[['DocumentKey', 'CustomerKey', 'DocumentDate', 'DueDate', 'ClearingDate', 'InvoicedAmount']].dropna()
    print('Total size (dropna filter):\t\t\t', len(df))

    for col in ['DocumentKey']:
        df[col] = df[col].apply(lambda x: x.split('|')[2])

    for col in ['DocumentDate', 'DueDate', 'ClearingDate']:
        df[col] = pd.to_datetime(df[col])

    for col in ['DocumentKey', 'CustomerKey', 'InvoicedAmount']:
        df[col] = pd.to_numeric(df[col])

    df = df[(df['DocumentDate'] < df['DueDate']) & (df['DocumentDate'] < df['ClearingDate'])]
    df = df[df['InvoicedAmount'] >= 1000]

    print('Total size (Date and Invoice Amount filters):\t', len(df), '\n')

    return df


def step1_organize(filename):

    def group(x): return x.groupby(['CustomerKey', 'DueDate'])
    def order(x): return x.sort_values(by=['DueDate'], ascending=True, ignore_index=True)
    def merge(x1, x2): return pd.merge(order(x1), order(x2), how='left', on=['CustomerKey', 'DueDate'])

    df = pd.read_csv(filename, parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

    df['OS'] = ((df['ClearingDate'] - (df['DueDate'] - pd.to_timedelta(arg=df['DueDate'].dt.weekday, unit='D')))
                .astype('timedelta64[D]') > 0).astype(int)

    df_doc = order(group(df)['DocumentDate'].min().reset_index())
    df_cle = order(group(df)['ClearingDate'].max().reset_index())

    df_gp = pd.DataFrame({
        'CustomerKey': df_doc['CustomerKey'].values,
        'DocumentDate': df_doc['DocumentDate'].values,
        'DueDate': df_doc['DueDate'].values,
        'ClearingDate': df_cle['ClearingDate'].values,
    })

    df_gp = merge(df_gp, group(df).size().reset_index(name='InvoiceCount'))
    df_gp = merge(df_gp, group(df[df['OS'] == 1]).size().reset_index(name='OSInvoiceCount'))

    df_gp['OSInvoiceCount'] = df_gp['OSInvoiceCount'].fillna(0).astype(int)

    df_gp['R_OSInvoiceCount'] = df_gp['OSInvoiceCount'] / df_gp['InvoiceCount']
    df_gp['R_OSInvoiceCount'] = df_gp['R_OSInvoiceCount'].fillna(0)

    df_gp = merge(df_gp, group(df)['InvoicedAmount'].sum().reset_index(name='InvoiceAmount'))
    df_gp = merge(df_gp, group(df[df['OS'] == 1])['InvoicedAmount'].sum().reset_index(name='OSInvoiceAmount'))

    df_gp['OSInvoiceAmount'] = df_gp['OSInvoiceAmount'].fillna(0)

    df_gp['R_OSInvoiceAmount'] = df_gp['OSInvoiceAmount'] / df_gp['InvoiceAmount']
    df_gp['R_OSInvoiceAmount'] = df_gp['R_OSInvoiceAmount'].fillna(0)

    return df_gp


def step2_generate(filename):
    df = pd.read_csv(filename, parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

    df['DaysToEndMonth'] = ((df['DueDate'] + pd.offsets.MonthEnd(0)) - df['DueDate']).dt.days
    df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).dt.days).clip(lower=0)
    df['DaysLateAM'] = (df['DaysLate'] - df['DaysToEndMonth']).clip(lower=0)

    df['PaymentCategory'] = 0
    df.loc[(df['DaysLate'] > 0) & (df['DaysLateAM'] <= 0), 'PaymentCategory'] = 1
    df.loc[(df['DaysLate'] > 0) & (df['DaysLateAM'] > 0), 'PaymentCategory'] = 2

    features = []
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for x in tqdm(pool.imap(partial(_historic), df[['CustomerKey', 'DueDate']].values), total=len(df)):
            features.append(x)
        pool.close()
        pool.join()

    df = pd.merge(df, pd.DataFrame(features), how='left', on=['CustomerKey', 'DueDate'])
    df.drop(['DaysToEndMonth', 'DaysLate', 'DaysLateAM', 'PaymentCategory'], axis=1, inplace=True)

    return df


def step3_prepare(filename):
    df = pd.read_csv(filename, parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

    df['DaysToDueDate'] = (df['DueDate'] - df['DocumentDate']).dt.days
    df['DaysToEndMonth'] = ((df['DueDate'] + pd.offsets.MonthEnd(0)) - df['DueDate']).dt.days

    df['WeekdayEndMonth'] = (df['DueDate'] + pd.offsets.MonthEnd(0)).dt.weekday

    df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).dt.days).clip(lower=0)
    df['DaysLateAM'] = (df['DaysLate'] - df['DaysToEndMonth']).clip(lower=0)

    df['PaidLate'] = 0
    df.loc[df['DaysLate'] > 0, 'PaidLate'] = 1

    df['PaidLateAM'] = 0
    df.loc[df['DaysLateAM'] > 0, 'PaidLateAM'] = 1

    df['PaymentCategory'] = 0
    df.loc[(df['DaysLate'] > 0) & (df['DaysLateAM'] <= 0), 'PaymentCategory'] = 1
    df.loc[(df['DaysLate'] > 0) & (df['DaysLateAM'] > 0), 'PaymentCategory'] = 2

    dfs = df.drop(['CustomerKey', 'DocumentDate', 'ClearingDate', 'DueDate'], axis=1)
    dfs['DueDate'] = df['DueDate']

    return df


def _historic(x, window_size=120):
    due_date = pd.to_datetime(x[1])

    end_due_date = pd.to_datetime(due_date - pd.to_timedelta(arg=due_date.weekday(), unit='D'))
    start_due_date = end_due_date - pd.DateOffset(months=window_size)

    c_data = df[df['CustomerKey'] == x[0]]

    h = c_data[(c_data['DueDate'] >= start_due_date) & (c_data['DueDate'] < end_due_date)]
    h_late = h[h['PaymentCategory'] == 1]
    h_very_late = h[h['PaymentCategory'] == 2]

    o = c_data[(c_data['DocumentDate'] >= start_due_date) & (c_data['DocumentDate'] < end_due_date) &
               (c_data['ClearingDate'] > end_due_date) & (c_data['DueDate'] != due_date)]
    o_late = o[o['PaymentCategory'] == 1]
    o_very_late = o[o['PaymentCategory'] == 2]

    with np.errstate(divide='ignore', invalid='ignore'):
        features = {
            'CustomerKey': x[0],
            'DueDate': due_date,

            'MAD_DaysLate': h_late['DaysLate'].mad(),
            'MED_DaysLate': h_late['DaysLate'].median(),

            'MAD_DaysLateAM': h_very_late['DaysLate'].mad(),
            'MED_DaysLateAM': h_very_late['DaysLate'].median(),


            'MAD_OSDaysLate': o_late['DaysLate'].mad(),
            'MED_OSDaysLate': o_late['DaysLate'].median(),

            'MAD_OSDaysLateAM': o_very_late['DaysLate'].mad(),
            'MED_OSDaysLateAM': o_very_late['DaysLate'].median(),


            'PaidCount': h['InvoiceCount'].sum(),
            'PaidLateCount': h_late['InvoiceCount'].sum(),
            'PaidLateAMCount': h_very_late['InvoiceCount'].sum(),

            'R_PaidLateCount': np.true_divide(h_late['InvoiceCount'].sum(), h['InvoiceCount'].sum()),
            'R_PaidLateAMCount': np.true_divide(h_very_late['InvoiceCount'].sum(), h['InvoiceCount'].sum()),


            'PaidAmount': h['InvoiceAmount'].sum(),
            'PaidLateAmount': h_late['InvoiceAmount'].sum(),
            'PaidLateAMAmount': h_very_late['InvoiceAmount'].sum(),

            'R_PaidLateAmount': np.true_divide(h_late['InvoiceAmount'].sum(), h['InvoiceAmount'].sum()),
            'R_PaidLateAMAmount': np.true_divide(h_very_late['InvoiceAmount'].sum(), h['InvoiceAmount'].sum()),


            'OSCount': o['InvoiceCount'].sum(),
            'OSLateCount': o_late['InvoiceCount'].sum(),
            'OSLateAMCount': o_very_late['InvoiceCount'].sum(),

            'R_OSLateCount': np.true_divide(o_late['InvoiceCount'].sum(), o['InvoiceCount'].sum()),
            'R_OSLateAMCount': np.true_divide(o_very_late['InvoiceCount'].sum(), o['InvoiceCount'].sum()),


            'OSAmount': o['InvoiceAmount'].sum(),
            'OSLateAmount': o_late['InvoiceAmount'].sum(),
            'OSLateAMAmount': o_very_late['InvoiceAmount'].sum(),

            'R_OSLateAmount': np.true_divide(o_late['InvoiceAmount'].sum(), o['InvoiceAmount'].sum()),
            'R_OSLateAMAmount': np.true_divide(o_very_late['InvoiceAmount'].sum(), o['InvoiceAmount'].sum()),
        }

    for k in features.keys():
        features[k] = np.nan_to_num(features[k], nan=0.0, posinf=0.0, neginf=0.0)

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int)

    arg = parser.parse_args()
    pathin = os.path.join('..', 'data')

    if arg.step == 0:
        df = pd.concat([
            step0_extract(os.path.join(pathin, 'base0a.csv')),
            step0_extract(os.path.join(pathin, 'base0b.csv')),
            step0_extract(os.path.join(pathin, 'base0c.csv')),
        ])

    elif arg.step == 1:
        df = step1_organize(os.path.join(pathin, 'base1.csv'))

    elif arg.step == 2:
        df = step2_generate(os.path.join(pathin, 'base2.csv'))

    elif arg.step == 3:
        df = step3_prepare(os.path.join(pathin, 'base3.csv'))

    print('Final size', len(df))
    df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
    df.to_csv(f'./data/base{arg.step + 1}.csv', index=False)
