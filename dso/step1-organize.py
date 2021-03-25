import pandas as pd


def group(x):
    return x.groupby(['CustomerKey', 'DueDate'])


def order(x):
    return x.sort_values(by=['DueDate'], ascending=True, ignore_index=True)


def merge(x1, x2):
    return pd.merge(order(x1), order(x2), how='left', on=['CustomerKey', 'DueDate'])


df = pd.read_csv('./data/base1.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

df_doc = order(group(df)['DocumentDate'].min().reset_index())
df_cle = order(group(df)['ClearingDate'].max().reset_index())

df_gp = pd.DataFrame({
    'CustomerKey': df_doc['CustomerKey'].values,
    'DocumentDate': df_doc['DocumentDate'].values,
    'DueDate': df_doc['DueDate'].values,
    'ClearingDate': df_cle['ClearingDate'].values,
})

df['PaidLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]') > 0).astype(int)

df_gp = merge(df_gp, group(df).size().reset_index(name='Count'))
df_gp = merge(df_gp, group(df[df['PaidLate'] == 1]).size().reset_index(name='OSCount'))

df_gp['OSCount'] = df_gp['OSCount'].fillna(0).astype(int)

df_gp['OSCountRatio'] = df_gp['OSCount'] / df_gp['Count']
df_gp['OSCountRatio'] = df_gp['OSCountRatio'].fillna(0)

df_gp = merge(df_gp, group(df)['InvoicedAmount'].sum().reset_index(name='Amount'))
df_gp = merge(df_gp, group(df[df['PaidLate'] == 1])['InvoicedAmount'].sum().reset_index(name='OSAmount'))

df_gp['OSAmount'] = df_gp['OSAmount'].fillna(0)

df_gp['OSAmountRatio'] = df_gp['OSAmount'] / df_gp['Amount']
df_gp['OSAmountRatio'] = df_gp['OSAmountRatio'].fillna(0)

print('Final size', len(df_gp))

df_gp.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df_gp.to_csv('./data/base2.csv', index=False)
