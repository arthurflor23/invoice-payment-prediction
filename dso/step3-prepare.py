import numpy as np
import pandas as pd


def setup_bins(df, col, bins, sufix=''):
    bins = [-np.inf] + bins + [np.inf]
    labels = [f'{bins[i]} - {bins[i+1]-1}' for i in range(len(bins[:-1]))]

    b_col = f'{col}{sufix}'
    l_col = f'{col}{sufix}Label'

    df[l_col] = pd.cut(df[col], bins=bins, labels=labels, right=False, include_lowest=True)
    df[[b_col]] = df[[l_col]].apply(lambda x: pd.Categorical(x, ordered=True).codes)

    return df


df = pd.read_csv('./data/base3.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysToClearing'] = (df['ClearingDate'] - df['DueDate']).dt.days

df = setup_bins(df, col='DaysToClearing', bins=[1], sufix='Bin')
df = setup_bins(df, col='DaysToClearing', bins=[1, 8, 15, 22, 29], sufix='Buc')

df.drop(['CustomerKey',
         'DocumentDate',
         'ClearingDate',
         'DueDate',
         'DaysToClearing',
         'DaysToClearingBinLabel',
         'DaysToClearingBucLabel'], axis=1, inplace=True)

df.to_csv('./data/base4.csv', index=False)
