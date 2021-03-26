from tqdm import tqdm
from functools import partial

import multiprocessing
import numpy as np
import pandas as pd


def fx(x, window_size=120):
    doc_date = pd.to_datetime(x[1])
    due_date = pd.to_datetime(x[2])
    cle_date = pd.to_datetime(x[3])

    c_data = df[df['CustomerKey'] == x[0]]

    past_due_date = due_date - pd.DateOffset(months=window_size)

    historic = c_data[(c_data['DueDate'] >= past_due_date) &
                      (c_data['DueDate'] < due_date)]
    historic_late = historic[historic['PaidLate'] == 1]

    outstanding = c_data[(c_data['DocumentDate'] >= past_due_date) &
                         (c_data['DocumentDate'] < due_date) &
                         (c_data['ClearingDate'] >= due_date)]
    outstanding_late = outstanding[outstanding['PaidLate'] == 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        features = {
            'CustomerKey': x[0],
            'DueDate': due_date,

            'DaysLateMad': historic['DaysLate'].mad(),
            'OSInvoiceDaysLateMad': outstanding['DaysLate'].mad(),

            'DaysLateMedian': historic['DaysLate'].median(),
            'OSInvoiceDaysLateMedian': outstanding['DaysLate'].median(),

            'TermDays': (due_date - doc_date).days,

            'PaidCount': historic['Count'].sum(),
            'PaidLateCount': historic_late['Count'].sum(),
            'PaidLateCountRatio': np.true_divide(historic_late['Count'].sum(), historic['Count'].sum()),

            'PaidAmount': historic['Amount'].sum(),
            'PaidLateAmount': historic_late['Amount'].sum(),
            'PaidLateAmountRatio': np.true_divide(historic_late['Amount'].sum(), historic['Amount'].sum()),

            'OSInvoiceCount': outstanding['Count'].sum(),
            'OSInvoiceLateCount': outstanding_late['Count'].sum(),
            'OSInvoiceLateCountRatio': np.true_divide(outstanding_late['Count'].sum(), outstanding['Count'].sum()),

            'OSInvoiceAmount': outstanding['Amount'].sum(),
            'OSInvoiceLateAmount': outstanding_late['Amount'].sum(),
            'OSInvoiceLateAmountRatio': np.true_divide(outstanding_late['Amount'].sum(), outstanding['Amount'].sum()),

            'PaymentDays': (cle_date - due_date).days,
        }

    for k in features.keys():
        features[k] = np.nan_to_num(features[k], nan=0.0, posinf=0.0, neginf=0.0)

    return features


df = pd.read_csv('./data/base2.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]')).astype(int)
df['PaidLate'] = (df['DaysLate'] > 0).astype(int)

features = []
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    cols = ['CustomerKey', 'DocumentDate', 'DueDate', 'ClearingDate']

    for x in tqdm(pool.imap(partial(fx), df[cols].values), total=len(df)):
        features.append(x)
    pool.close()
    pool.join()

df = pd.merge(df, pd.DataFrame(features), how='left', on=['CustomerKey', 'DueDate'])
df.drop(['PaidLate', 'DaysLate'], axis=1, inplace=True)

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base3.csv', index=False)