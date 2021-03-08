from statistics import mode
import pandas as pd


def group(x): return x.groupby(['CustomerKey', 'DueDate'])
def order(x): return x.sort_values(by=['DueDate'], ascending=True, ignore_index=True)
def merge(x1, x2): return pd.merge(order(x1), order(x2), how='left', on=['CustomerKey', 'DueDate'])


df = pd.read_csv('./data/base1.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)
df['PaidLate'] = ((df['ClearingDate'] - df['DueDate']).astype('timedelta64[D]') > 0).astype(int)

df1 = order(group(df)['DocumentDate'].min().reset_index())
df2 = order(group(df)['ClearingDate'].max().reset_index())

dfs = pd.DataFrame({
    'CustomerKey': df1['CustomerKey'].values,
    'DocumentDate': df1['DocumentDate'].values,
    'DueDate': df1['DueDate'].values,
    'ClearingDate': df2['ClearingDate'].values
})


dfs = merge(dfs, group(df)['DocumentType'].agg(lambda x: len(list(set(x)))).reset_index(name="TypeCount"))
dfs = merge(dfs, group(df)['DocumentType'].agg(mode).reset_index(name="Type"))

dfs = merge(dfs, group(df)['PaymentTerms'].agg(lambda x: len(list(set(x)))).reset_index(name="TermsCount"))
dfs = merge(dfs, group(df)['PaymentTerms'].agg(mode).reset_index(name="Terms"))


dfs = merge(dfs, group(df).size().reset_index(name='Count'))
dfs = merge(dfs, group(df[df['PaidLate'] == 1]).size().reset_index(name='OutstandingCount'))

dfs = dfs.fillna(0)
dfs['OutstandingCount'] = dfs['OutstandingCount'].astype(int)
dfs['OutstandingCountRatio'] = dfs['OutstandingCount'] / dfs['Count']


dfs = merge(dfs, group(df)['InvoicedAmount'].sum().reset_index(name='Amount'))
dfs = merge(dfs, group(df[df['PaidLate'] == 1])['InvoicedAmount'].sum().reset_index(name='OutstandingAmount'))

dfs = dfs.fillna(0)
dfs['OutstandingAmountRatio'] = dfs['OutstandingAmount'] / dfs['Amount']


dfs.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
dfs.to_csv('./data/base2a.csv', index=False)
