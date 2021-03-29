import pandas as pd


def load_dataset(filename):
    df = pd.read_csv(filename, low_memory=False)
    print('Total size:\t\t\t\t\t', len(df))

    df = df[df['TransactionType'] == 'FI-InvoicedDocument']
    print('Total size (transaction type filter):\t\t', len(df))

    df = df[[
        'DocumentKey',
        'CustomerKey',
        'DocumentDate',
        'DueDate',
        'ClearingDate',
        'InvoicedAmount',
    ]].dropna()

    print('Total size (dropna filter):\t\t\t', len(df))

    for col in ['DocumentKey']:
        df[col] = df[col].apply(lambda x: x.split('|')[2])

    for col in ['DocumentDate', 'DueDate', 'ClearingDate']:
        df[col] = pd.to_datetime(df[col])

    for col in ['DocumentKey', 'CustomerKey', 'InvoicedAmount']:
        df[col] = pd.to_numeric(df[col])

    df = df[(df['DocumentDate'] < df['DueDate']) & (df['DocumentDate'] < df['ClearingDate'])]
    df = df[df['InvoicedAmount'] >= 1000]

    print('Total size (Date and Invoice Amount filters):\t', len(df), '\n')

    return df


df1 = load_dataset('./data/base0a.csv')
df2 = load_dataset('./data/base0b.csv')
df3 = load_dataset('./data/base0c.csv')
df = pd.concat([df1, df2, df3])

print('Final size', len(df))

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base1.csv', index=False)
