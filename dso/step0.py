import numpy as np
import pandas as pd


def load_dataset(filename):
    df = pd.read_csv(filename, low_memory=False)

    print("Total size:")
    print(df.info(), '\n\n')

    # Filter transaction type equal to invoiced
    df = df[df['TransactionType'] == 'FI-InvoicedDocument']

    print("Total size (transaction type filter):")
    print(df.info(), '\n\n')

    # Select columns of interest
    df = df[[
        'DocumentKey',
        'CustomerKey',
        'DocumentDate',
        'DueDate',
        'ClearingDate',
        'DocumentType',
        'PaymentTerms',
        'InvoicedAmount',
    ]].dropna()

    print("Total size (dropna filter):")
    print(df.info(), '\n\n')

    # Cast dTypes of the columns
    for col in ['DocumentKey']:
        df[col] = df[col].apply(lambda x: x.split('|')[2] if len(x.split('|')) == 3 else np.nan)

    for col in ['DocumentDate', 'DueDate', 'ClearingDate']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['DocumentType', 'PaymentTerms']:
        df[col] = df[col].apply(lambda x: int(''.join(format(ord(w), '') for w in str(x))))

    for col in ['DocumentKey', 'CustomerKey', 'InvoicedAmount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


df1 = load_dataset('./data/base0a.csv')
df2 = load_dataset('./data/base0b.csv')
df = pd.concat([df1, df2])

df = df[(df['DocumentDate'] < df['DueDate']) & (df['DocumentDate'] < df['ClearingDate'])]
df = df[df['InvoicedAmount'] >= 1000]

df.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
df.to_csv('./data/base1.csv', index=False)
