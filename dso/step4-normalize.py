from sklearn.preprocessing import RobustScaler

import pandas as pd

df = pd.read_csv('./data/base4.csv', low_memory=False)

qt = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=True)
df.iloc[:, 0:-2] = qt.fit_transform(df.iloc[:, 0:-2].to_numpy())

print(df.info(), '\n\n', df)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('./data/base5.csv', index=False)
