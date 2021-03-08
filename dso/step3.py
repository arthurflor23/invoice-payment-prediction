import numpy as np
import pandas as pd

df = pd.read_csv('./data/base2b.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)
df = df[[c for c in df.columns if c != 'DueDate'] + ['DueDate']]

df['DaysLate'] = (df['ClearingDate'] - df['DueDate']).dt.days.clip(lower=0)

df['PaidLateLabel'] = pd.cut(df['DaysLate'],
                             bins=[-np.inf, 1, np.inf],
                             labels=['ontime', 'late'],
                             right=False,
                             include_lowest=True)

df['PaidLate'] = (df['DaysLate'] > 0).astype('int8')


df['DaysLateBucketLabel'] = pd.cut(df['DaysLate'],
                                   bins=[-np.inf, 1, 8, 15, 22, 28, np.inf],
                                   labels=['ontime', '1-7', '8-14', '15-21', '22-28', '29+'],
                                   right=False,
                                   include_lowest=True)

df[['DaysLateBucket']] = df[['DaysLateBucketLabel']].apply(lambda x: pd.Categorical(x, ordered=True).codes)

df.drop(['CustomerKey', 'DocumentDate', 'ClearingDate', 'DaysLate'], axis=1, inplace=True)

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base3.csv', index=False)
