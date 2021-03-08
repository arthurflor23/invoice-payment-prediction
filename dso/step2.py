from tqdm import tqdm
from functools import partial

import multiprocessing
import numpy as np
import pandas as pd


def fx(x, w=2):
    due_date = pd.to_datetime(x[1])
    due_date_weekday = pd.DatetimeIndex([due_date]).weekday.values[0]
    past_due_date = due_date - pd.DateOffset(months=w)

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

            'AvgDaysLate': historic['DaysLate'].median(),
            'StdDaysLate': historic['DaysLate'].std(),

            'AvgDaysOutstandingLate': outstanding['DaysLate'].median(),
            'StdDaysOutstandingLate': outstanding['DaysLate'].std(),

            'PaidCountWeekday': weekday['Count'].sum(),
            'PaidLateCountWeekday': weekday_late['Count'].sum(),
            'PaidLateCountWeekdayRatio': np.true_divide(weekday_late['Count'].sum(), weekday['Count'].sum()),

            'PaidAmountWeekday': weekday['Amount'].sum(),
            'PaidLateAmountWeekday': weekday_late['Amount'].sum(),
            'PaidLateAmountWeekdayRatio': np.true_divide(weekday_late['Amount'].sum(), weekday['Amount'].sum()),

            'PaidCountTotal': historic['Count'].sum(),
            'PaidLateCountTotal': historic_late['Count'].sum(),
            'PaidLateCountTotalRatio': np.true_divide(historic_late['Count'].sum(), historic['Count'].sum()),

            'PaidAmountTotal': historic['Amount'].sum(),
            'PaidLateAmountTotal': historic_late['Amount'].sum(),
            'PaidLateAmountTotalRatio': np.true_divide(historic_late['Amount'].sum(), historic['Amount'].sum()),

            'OutstandingCountTotal': outstanding['Count'].sum(),
            'OutstandingLateCountTotal': outstanding_late['Count'].sum(),
            'OutstandingLateCountTotalRatio': np.true_divide(outstanding_late['Count'].sum(), outstanding['Count'].sum()),

            'OutstandingAmountTotal': outstanding['Amount'].sum(),
            'OutstandingLateAmountTotal': outstanding_late['Amount'].sum(),
            'OutstandingLateAmountTotalRatio': np.true_divide(outstanding_late['Amount'].sum(), outstanding['Amount'].sum())
        }

    for k in features.keys():
        features[k] = np.nan_to_num(features[k], nan=0.0, posinf=0.0, neginf=0.0)

    return features


df = pd.read_csv('./data/base2a.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]')).astype(int).clip(lower=0)
df['PaidLate'] = (df['DaysLate'] > 0).astype(int)

df['DaysToDueDate'] = ((df['DueDate'] - df['DocumentDate']).astype('timedelta64[D]')).astype(int)
df['DueDateWeekDay'] = pd.DatetimeIndex(df['DueDate']).weekday

features = []
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    for x in tqdm(pool.imap(partial(fx, w=2), df[['CustomerKey', 'DueDate']].values), total=len(df)):
        features.append(x)
    pool.close()
    pool.join()

df = pd.merge(df, pd.DataFrame(features), how='left', on=['CustomerKey', 'DueDate'])
df.drop(['PaidLate', 'DaysLate'], axis=1, inplace=True)

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base2b.csv', index=False)
