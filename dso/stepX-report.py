from pandas_profiling import ProfileReport

import pandas as pd

df = pd.read_csv('./data/base5.csv', low_memory=False)

profile = ProfileReport(df.drop(['DaysToClearingBin', 'DaysToClearingBuc'], axis=1),
                        title='Pandas Profiling Report', explorative=True)

profile.to_file("base_report.html")
