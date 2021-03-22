from tqdm import tqdm
from functools import partial

import multiprocessing
import numpy as np
import pandas as pd


def fx(x, window_size=120):
    due_date = pd.to_datetime(x[1])
    due_date_weekday = pd.DatetimeIndex([due_date]).weekday.values[0]
    past_due_date = due_date - pd.DateOffset(months=window_size)

    c_data = df[df['CustomerKey'] == x[0]]

    historic = c_data[(c_data['DueDate'] >= past_due_date) &
                      (c_data['DueDate'] < due_date)]
    historic_late = historic[historic['PaidLate'] == 1]

    weekday = historic[historic['DueDateWeekDay'] == due_date_weekday]
    weekday_late = weekday[weekday['PaidLate'] == 1]

    outstanding = c_data[(c_data['DocumentDate'] >= past_due_date) &
                         (c_data['DocumentDate'] < due_date) &
                         (c_data['ClearingDate'] >= due_date)]
    outstanding_late = outstanding[outstanding['PaidLate'] == 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        features = {
            'CustomerKey': x[0],
            'DueDate': due_date,

            'DaysToDueDateMD': historic['DaysToDueDate'].median(),

            'WDPaidCount': weekday['InvoiceCount'].sum(),
            'WDPaidLateCount': weekday_late['InvoiceCount'].sum(),
            'WDPaidLateCountR': np.true_divide(weekday_late['InvoiceCount'].sum(),
                                               weekday['InvoiceCount'].sum()),

            'WDPaidAmount': weekday['InvoiceAmount'].sum(),
            'WDPaidLateAmount': weekday_late['InvoiceAmount'].sum(),
            'WDPaidLateAmountR': np.true_divide(weekday_late['InvoiceAmount'].sum(),
                                                weekday['InvoiceAmount'].sum()),

            'WDDaysLateMD': weekday['DaysLate'].median(),

            'TTPaidCount': historic['InvoiceCount'].sum(),
            'TTPaidLateCount': historic_late['InvoiceCount'].sum(),
            'TTPaidLateCountR': np.true_divide(historic_late['InvoiceCount'].sum(),
                                               historic['InvoiceCount'].sum()),

            'TTPaidAmount': historic['InvoiceAmount'].sum(),
            'TTPaidLateAmount': historic_late['InvoiceAmount'].sum(),
            'TTPaidLateAmountR': np.true_divide(historic_late['InvoiceAmount'].sum(),
                                                historic['InvoiceAmount'].sum()),

            'TTDaysLateMD': historic['DaysLate'].median(),

            'TTOSCount': outstanding['InvoiceCount'].sum(),
            'TTOSLateCount': outstanding_late['InvoiceCount'].sum(),
            'TTOSLateCountR': np.true_divide(outstanding_late['InvoiceCount'].sum(),
                                             outstanding['InvoiceCount'].sum()),

            'TTOSAmount': outstanding['InvoiceAmount'].sum(),
            'TTOSLateAmount': outstanding_late['InvoiceAmount'].sum(),
            'TTOSLateAmountR': np.true_divide(outstanding_late['InvoiceAmount'].sum(),
                                              outstanding['InvoiceAmount'].sum()),

            'TTOSDaysLateMD': outstanding['DaysLate'].median(),
        }

    for k in features.keys():
        features[k] = np.nan_to_num(features[k], nan=0.0, posinf=0.0, neginf=0.0)

    return features


df = pd.read_csv('./data/base2.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]')).astype(int).clip(lower=0)
df['PaidLate'] = (df['DaysLate'] > 0).astype(int)

df['DaysToDueDate'] = ((df['DueDate'] - df['DocumentDate']).astype('timedelta64[D]')).astype(int)
df['DueDateWeekDay'] = pd.DatetimeIndex(df['DueDate']).weekday

features = []
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    for x in tqdm(pool.imap(partial(fx), df[['CustomerKey', 'DueDate']].values), total=len(df)):
        features.append(x)
    pool.close()
    pool.join()

df = pd.merge(df, pd.DataFrame(features), how='left', on=['CustomerKey', 'DueDate'])
df.drop(['PaidLate', 'DaysLate', 'DueDateWeekDay'], axis=1, inplace=True)

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base3.csv', index=False)
