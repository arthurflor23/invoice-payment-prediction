import pandas as pd


def group(x):
    return x.groupby(['CustomerKey', 'DueDate'])


def order(x):
    return x.sort_values(by=['DueDate'], ascending=True, ignore_index=True)


def merge(x1, x2):
    return pd.merge(order(x1), order(x2), how='left', on=['CustomerKey', 'DueDate'])


df = pd.read_csv('./data/base1.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)
df['PaidLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]') > 0).astype(int)

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

print('Final size', len(df_gp))

df_gp.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df_gp.to_csv('./data/base2.csv', index=False)
