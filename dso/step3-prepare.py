import pandas as pd


df = pd.read_csv('./data/base3.csv', parse_dates=['DocumentDate', 'DueDate', 'ClearingDate'], low_memory=False)

vip_customers = [305689, 377986, 380520, 400021, 400022, 400074, 400120, 400127, 400129, 400250, 400546, 400124,
                 7398, 71954, 72052, 73101, 73311, 74935, 76085, 77890, 78190, 78456, 78457, 78475, 79110, 72070,
                 79379, 79717, 301827, 330695, 378803, 378804, 378805, 379986, 380092, 380700, 380701, 380763, 300860,
                 380956, 381086, 383387, 385891, 386621, 388651, 389233, 391526, 391527, 391528, 393444, 395292, 395948]

df['PartnerCustomer'] = 0
df.loc[df['CustomerKey'].isin(vip_customers), 'PartnerCustomer'] = 1

df['DaysToDueDate'] = (df['DueDate'] - df['DocumentDate']).dt.days
df['DaysToEndMonth'] = ((df['DueDate'] + pd.offsets.MonthEnd(0)) - df['DueDate']).dt.days


df['DaysLate'] = ((df['ClearingDate'] - df['DueDate']).dt.days).clip(lower=0)
df['DaysLateAfterMonth'] = (df['DaysLate'] - df['DaysToEndMonth']).clip(lower=0)

df['PaymentCategory'] = 0
df.loc[(df['DaysLate'] > 0) & (df['DaysLateAfterMonth'] <= 0), 'PaymentCategory'] = 1
df.loc[(df['DaysLate'] > 0) & (df['DaysLateAfterMonth'] > 0), 'PaymentCategory'] = 2


dfs = df.drop(['CustomerKey', 'DocumentDate', 'ClearingDate', 'DueDate'], axis=1)
dfs['DueDate'] = df['DueDate']

dfs.sort_values(by=['DueDate'], ascending=True, ignore_index=True, inplace=True)
dfs.to_csv('./data/base4.csv', index=False)
