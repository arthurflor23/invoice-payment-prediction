from tqdm import tqdm
from functools import partial

import multiprocessing
import numpy as np
import pandas as pd


def fx(x, window_size=120):
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


df = pd.read_csv('./data/base2.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysToEndMonth'] = ((df['DueDate'] + pd.offsets.MonthEnd(0)) - df['DueDate']).dt.days
df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).dt.days).clip(lower=0)
df['DaysLateAM'] = (df['DaysLate'] - df['DaysToEndMonth']).clip(lower=0)

df['PaymentCategory'] = 0
df.loc[(df['DaysLate'] > 0) & (df['DaysLateAM'] <= 0), 'PaymentCategory'] = 1
df.loc[(df['DaysLate'] > 0) & (df['DaysLateAM'] > 0), 'PaymentCategory'] = 2

features = []
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    for x in tqdm(pool.imap(partial(fx), df[['CustomerKey', 'DueDate']].values), total=len(df)):
        features.append(x)
    pool.close()
    pool.join()

df = pd.merge(df, pd.DataFrame(features), how='left', on=['CustomerKey', 'DueDate'])
df.drop(['DaysToEndMonth', 'DaysLate', 'DaysLateAM', 'PaymentCategory'], axis=1, inplace=True)

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base3.csv', index=False)
