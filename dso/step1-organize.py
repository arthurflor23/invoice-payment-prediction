from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def setup_bins(df, col, bins, sufix=''):
    bins = [-np.inf] + bins + [np.inf]
    labels = [f'{bins[i]} - {bins[i+1]-1}' for i in range(len(bins[:-1]))]

    b_col = f'{col}{sufix}'
    l_col = f'{col}{sufix}Label'

    df[l_col] = pd.cut(df[col], bins=bins, labels=labels, right=False, include_lowest=True)
    df[[b_col]] = df[[l_col]].apply(lambda x: pd.Categorical(x, ordered=True).codes)

    return df


def group(x):
    return x.groupby(['CustomerKey', 'DueDate'])


def order(x):
    return x.sort_values(by=['DueDate'], ascending=True, ignore_index=True)


def merge(x1, x2):
    return pd.merge(order(x1), order(x2), how='left', on=['CustomerKey', 'DueDate'])


def outliers_quantile(df, col, q_low=0, q_hi=1):
    return df[(df[col].quantile(q_low) < df[col]) & (df[col] < df[col].quantile(q_hi))]


def outliers_quantile_groupby(df, col, q_low=0, q_hi=1):
    return df[df.groupby('DaysToClearingDateBuc')[col].transform(lambda x: (x.quantile(q_low) < x) & (x < x.quantile(q_hi))).eq(1)]


def outliers_zscore(df, col, z=3):
    return df[np.abs(zscore(df[col])) < z]


def outliers_zscore_groupby(df, col, z=3):
    return df[df.groupby('DaysToClearingDateBuc')[col].transform(lambda x: np.abs(zscore(x)) < z).eq(1)]


def outliers_iqr(df, col, q_low=0, q_hi=1):
    Q1 = df[col].quantile(q_low)
    Q3 = df[col].quantile(q_hi)
    IQR = Q3 - Q1
    return df[((Q1 - 1.5 * IQR) < df[col]) & (df[col] < (Q3 + 1.5 * IQR))]


def outliers_iqr_groupby(df, col, q_low=0, q_hi=1):
    def rm_outliers(x):
        Q1 = x.quantile(q_low)
        Q3 = x.quantile(q_hi)
        IQR = Q3 - Q1
        return ((Q1 - 1.5 * IQR) < x) & (x < (Q3 + 1.5 * IQR))
    return df[df.groupby('DaysToClearingDateBuc')[col].transform(lambda x: rm_outliers(x)).eq(1)]


def outliers_isolation_groupby(df, col):
    df.reset_index(drop=True, inplace=True)
    df['iso'] = 1

    for i in df.groupby(['DaysToClearingDateBuc']).groups.keys():
        clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42)
        clf.fit(df[df['DaysToClearingDateBuc'] == i][col].values.reshape(-1, 1))
        df.loc[df['DaysToClearingDateBuc'] == i, 'iso'] = clf.predict(
            df[df['DaysToClearingDateBuc'] == i][col].values.reshape(-1, 1))

    df = df[df['iso'] == 1]
    df.drop('iso', axis=1, inplace=True)
    return df


def outliers_robust_zscore_groupby(df, col, group_col=None, z=3.5, mi=0.6745):
    df['rob'] = 0

    if group_col:
        for i in df.groupby([group_col]).groups.keys():
            median = df[df[group_col] == i][col].median()
            mad = df[df[group_col] == i][col].mad()
            df.loc[df[group_col] == i, 'rob'] = mi * (np.abs(df[df[group_col] == i][col] - median)) / mad
    else:
        df['rob'] = mi * (np.abs(df[col] - df[col].median())) / df[col].mad()

    df = df[df['rob'] <= z]
    df.drop('rob', axis=1, inplace=True)
    return df


df = pd.read_csv('./data/base1.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df['DaysToDueDate'] = ((df['DueDate'] - df['DocumentDate']).astype('timedelta64[D]')).astype(int)
df['PaidLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]') > 0).astype(int)
df['DaysToClearingDate'] = (df['ClearingDate'] - df['DueDate']).dt.days
df['DueDateMonth'] = pd.DatetimeIndex(df['DueDate']).month

vip_customers = [305689, 377986, 380520, 400021, 400022, 400074, 400120, 400127, 400129, 400250, 400546, 400124,
                 7398, 71954, 72052, 73101, 73311, 74935, 76085, 77890, 78190, 78456, 78457, 78475, 79110, 72070,
                 79379, 79717, 301827, 330695, 378803, 378804, 378805, 379986, 380092, 380700, 380701, 380763, 300860,
                 380956, 381086, 383387, 385891, 386621, 388651, 389233, 391526, 391527, 391528, 393444, 395292, 395948]

df = df[~df['CustomerKey'].isin(vip_customers)]
df = df[(df['InvoicedAmount'] >= 1000)]

df = setup_bins(df, col='DaysToClearingDate', bins=[1], sufix='Bin')
df = setup_bins(df, col='DaysToClearingDate', bins=[1, 8, 15, 22, 29], sufix='Buc')

df = outliers_robust_zscore_groupby(df, col='DaysToClearingDate', group_col='DaysToClearingDateBuc')
df = outliers_robust_zscore_groupby(df, col='DaysToDueDate', group_col='DaysToClearingDateBuc')
df = outliers_robust_zscore_groupby(df, col='InvoicedAmount', group_col='DaysToClearingDateBuc')

df_doc = order(group(df)['DocumentDate'].min().reset_index())
df_cle = order(group(df)['ClearingDate'].max().reset_index())

df_gp = pd.DataFrame({
    'CustomerKey': df_doc['CustomerKey'].values,
    'DocumentDate': df_doc['DocumentDate'].values,
    'DueDate': df_doc['DueDate'].values,
    'ClearingDate': df_cle['ClearingDate'].values,
})

df_gp = merge(df_gp, group(df).size().reset_index(name='InvoiceCount'))
df_gp = merge(df_gp, group(df[df['PaidLate'] == 1]).size().reset_index(name='OSInvoiceCount'))

df_gp['OSInvoiceCount'] = df_gp['OSInvoiceCount'].fillna(0).astype(int)

df_gp['OSInvoiceCountRatio'] = df_gp['OSInvoiceCount'] / df_gp['InvoiceCount']
df_gp['OSInvoiceCountRatio'] = df_gp['OSInvoiceCountRatio'].fillna(0)

df_gp = merge(df_gp, group(df)['InvoicedAmount'].sum().reset_index(name='InvoiceAmount'))
df_gp = merge(df_gp, group(df[df['PaidLate'] == 1])['InvoicedAmount'].sum().reset_index(name='OSInvoiceAmount'))

df_gp['OSInvoiceAmount'] = df_gp['OSInvoiceAmount'].fillna(0)

df_gp['OSInvoiceAmountRatio'] = df_gp['OSInvoiceAmount'] / df_gp['InvoiceAmount']
df_gp['OSInvoiceAmountRatio'] = df_gp['OSInvoiceAmountRatio'].fillna(0)

df_gp['DaysToClearingDate'] = (df_gp['ClearingDate'] - df_gp['DueDate']).dt.days
df_gp = setup_bins(df_gp, col='DaysToClearingDate', bins=[1, 8, 15, 22, 29], sufix='Buc')
df_gp = outliers_robust_zscore_groupby(df_gp, col='InvoiceCount', group_col='DaysToClearingDateBuc')
df_gp = df_gp.drop(['DaysToClearingDate', 'DaysToClearingDateBucLabel', 'DaysToClearingDateBuc'], axis=1)

df_gp.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df_gp.to_csv('./data/base2.csv', index=False)
