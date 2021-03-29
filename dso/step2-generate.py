from tqdm import tqdm
from functools import partial

import multiprocessing
import numpy as np
import pandas as pd


def fx(x, window_size=120):
    due_date = pd.to_datetime(x[1])
    past_due_date = due_date - pd.DateOffset(months=window_size)

    c_data = df[df['CustomerKey'] == x[0]]

    historic = c_data[(c_data['DueDate'] >= past_due_date) &
                      (c_data['DueDate'] < due_date)]
    historic_late = historic[historic['PaidLate'] == 1]

    outstanding = c_data[(c_data['DocumentDate'] >= past_due_date) &
                         (c_data['DocumentDate'] < due_date) &
                         (c_data['ClearingDate'] > due_date) &
                         (c_data['DueDate'] != due_date)]
    outstanding_late = outstanding[outstanding['PaidLate'] == 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        features = {
            'CustomerKey': x[0],
            'DueDate': due_date,

            'MAD_DaysLate': historic['DaysLate'].mad(),
            'MAD_OSDaysLate': outstanding['DaysLate'].mad(),

            'MED_DaysLate': historic['DaysLate'].median(),
            'MED_OSDaysLate': outstanding['DaysLate'].median(),

            'PaidCount': historic['InvoiceCount'].sum(),
            'PaidLateCount': historic_late['InvoiceCount'].sum(),
            'R_PaidLateCount': np.true_divide(historic_late['InvoiceCount'].sum(), historic['InvoiceCount'].sum()),

            'PaidAmount': historic['InvoiceAmount'].sum(),
            'PaidLateAmount': historic_late['InvoiceAmount'].sum(),
            'R_PaidLateAmount': np.true_divide(historic_late['InvoiceAmount'].sum(), historic['InvoiceAmount'].sum()),

            'OSCount': outstanding['InvoiceCount'].sum(),
            'OSLateCount': outstanding_late['InvoiceCount'].sum(),
            'R_OSLateCount': np.true_divide(outstanding_late['InvoiceCount'].sum(), outstanding['InvoiceCount'].sum()),

            'OSAmount': outstanding['InvoiceAmount'].sum(),
            'OSLateAmount': outstanding_late['InvoiceAmount'].sum(),
            'R_OSLateAmount': np.true_divide(outstanding_late['InvoiceAmount'].sum(), outstanding['InvoiceAmount'].sum()),
        }

    for k in features.keys():
        features[k] = np.nan_to_num(features[k], nan=0.0, posinf=0.0, neginf=0.0)

    return features


df = pd.read_csv('./data/base2.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]')).astype(int)
df['PaidLate'] = (df['DaysLate'] > 0).astype(int)

features = []
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    for x in tqdm(pool.imap(partial(fx), df[['CustomerKey', 'DueDate']].values), total=len(df)):
        features.append(x)
    pool.close()
    pool.join()

df = pd.merge(df, pd.DataFrame(features), how='left', on=['CustomerKey', 'DueDate'])
df.drop(['PaidLate', 'DaysLate'], axis=1, inplace=True)

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base3.csv', index=False)
