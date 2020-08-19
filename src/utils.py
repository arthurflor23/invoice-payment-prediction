import datetime
import functools
import multiprocessing
import numpy as np
import pandas as pd
import os


class CSVManager():
    def __init__(self):
        self.df = None
        self.bins = None
        self.labels = None
        self.source = os.path.join('..', 'csv')
        self.target = os.path.join('..', 'csv-pp')

    def read(self, filename, sep=';', na_values=['N/I'], orderby=None, ascending=True):
        self.df = pd.read_csv(os.path.join(self.source, filename), sep=sep, na_values=na_values)

        if orderby is not None:
            self.df.sort_values(by=orderby, ascending=ascending, ignore_index=True, inplace=True)

    def save(self, filename, sep=';', na_rep=-1):
        os.makedirs(self.target, exist_ok=True)
        filepath = os.path.join(self.target, filename)
        self.df.to_csv(filepath, sep=sep, na_rep=na_rep, float_format='%g', index=False)

    def info(self, jupyter=True):
        if jupyter:
            return self.df.info()
        print(self.df.info())

    def head(self, value=10, jupyter=True):
        if jupyter:
            return self.df.head(value)
        print(self.df.head(value))

    def null_sum(self, jupyter=True):
        if jupyter:
            return self.df.isnull().sum()
        print(self.df.isnull().sum())

    def ppnan(self, dropna_cols=None, fillna_cols=None, fillna_value=None):
        if dropna_cols:
            self.df.dropna(subset=dropna_cols, inplace=True)
        if fillna_cols is not None and fillna_value is not None:
            self.df[fillna_cols] = self.df[fillna_cols].fillna(fillna_value)
        elif fillna_value is not None:
            self.df.fillna(value=fillna_value, inplace=True)

    def set_ratio(self, dividend_cols, divisor_cols):
        for dividend, divisor in zip(dividend_cols, divisor_cols):
            ratio_col = 'Ratio' + dividend + divisor
            self.df[ratio_col] = self.df[dividend] / self.df[divisor]
            self.df[ratio_col].fillna(0, inplace=True)

    def set_daysto(self, source_cols, target_cols):
        for src, tgt in zip(source_cols, target_cols):
            delta_col = 'DaysTo' + tgt
            self.df[delta_col] = self.df[tgt] - self.df[src]
            self.df[delta_col].fillna(pd.Timedelta(seconds=0), inplace=True)
            self.df[delta_col] = self.df[delta_col].astype('timedelta64[D]').astype(int)
            self.df[delta_col] = self.df[delta_col].clip(lower=0)

    def set_end_month(self, cols):
        for col in cols:
            self.df[col + 'EndMonth'] = self.df[col].apply(lambda x: x + pd.offsets.MonthEnd(1))

    def set_range(self, bins, cols):
        self.bins = bins + [np.inf]
        self.labels = [f'{self.bins[i]}-{self.bins[i+1]-1}' for i in range(len(self.bins[:-1]))]

        for col in cols:
            self.df[col + 'Range'] = pd.cut(self.df[col], bins=self.bins, labels=self.labels, right=False, include_lowest=True)
            self.df[[col + 'RangeCT']] = self.df[[col + 'Range']].apply(lambda x: pd.Categorical(x, ordered=True).codes)

    def extract_days(self, cols):
        for col in cols:
            self.df[col + 'Month'] = self.df[col].dt.month
            self.df[col + 'Day'] = self.df[col].dt.day
            self.df[col + 'WeekDay'] = self.df[col].dt.weekday

    def minmax_condition(self, min_cols, max_cols, filt=True):
        for mi, ma in zip(min_cols, max_cols):
            if filt:
                self.df = self.df[self.df[mi] <= self.df[ma]]
            else:
                self.df['After' + ma] = (self.df[mi] > self.df[ma]) * 1

    def cast_to_number(self, cols):
        self.df = self.df.apply(lambda x: [int(''.join(format(ord(w), '') for w in str(y)))
                                           if not str(y).isnumeric() else y for y in x] if x.name in cols else x)

    def cast_to_integer(self, cols):
        self.df = self.df.apply(lambda x: pd.to_numeric(x, downcast='integer') if x.name in cols else x)

    def cast_to_date(self, cols):
        self.df = self.df.apply(lambda x: pd.to_datetime(x, errors='coerce') if x.name in cols else x)

    def get_data_range(self, col, date, month_window):
        date_0 = pd.to_datetime(date)
        date_1 = date_0 - pd.DateOffset(months=month_window)
        date_2 = date_0 + pd.DateOffset(months=1)
        train = self.df[(self.df[col] >= date_1) & (self.df[col] < date_0)]
        test = self.df[(self.df[col] >= date_0) & (self.df[col] < date_2)]
        return train, test

    def get_df_binary_encoding(self, cols):
        df = self.df.copy()

        for col in cols:
            bincol = np.array([str('{0:b}'.format(x)) for x in df[col[1]].values])
            header = np.array([f'{col[1]}{i}' for i in range(col[0])])
            newcol = np.zeros((bincol.shape[0], col[0]), dtype=np.int8)

            for i in range(bincol.shape[0]):
                a = np.array(list(bincol[i]), dtype=np.int8)
                newcol[i][col[0] - len(a):] = a

            df2 = pd.DataFrame(newcol, columns=header)
            df.reset_index(drop=True, inplace=True)
            df = pd.concat([df, df2], axis=1)
            df.drop(columns=[col[1]], inplace=True)
        return df

    def calculate_per_bucket(self, bucket_col, amount_col, date_col, key_col, month_window):
        self.df.reset_index(drop=True, inplace=True)
        dflocal = self.df[[bucket_col, amount_col, date_col, key_col]].copy()

        if self.labels is None:
            self.labels = np.unique(dflocal[bucket_col])

        month_window = pd.DateOffset(months=month_window)
        arange_labels = np.arange(len(self.labels))

        min_month = dflocal[date_col].min()
        max_month = dflocal[date_col].max()
        one_month = pd.DateOffset(months=1)
        results = []

        while min_month <= max_month:
            records = dflocal[(dflocal[date_col] >= min_month - month_window) & (dflocal[date_col] < min_month + one_month)]
            curr_month = records[records[date_col] >= min_month].values

            batch = f'{min_month.year}-{min_month.month}/{max_month.year}-{max_month.month}'
            print(f'Preprocessing {curr_month.shape[0]} items ({batch})', end=' ')

            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                start_time = datetime.datetime.now()
                r = pool.map(functools.partial(self.apply_multiprocessing, records, arange_labels,
                                               month_window, bucket_col, amount_col, date_col, key_col), curr_month)
                print(f'~ {datetime.datetime.now() - start_time}')
                results.extend(r)
                pool.close()
                pool.join()

            min_month += one_month

        new_cols = np.array([[f'Bucket{i}Amount', f'Bucket{i}Count'] for i in arange_labels])
        results = pd.DataFrame(np.array(results), columns=new_cols.flatten())

        self.df.drop(labels=new_cols.flatten(), axis=1, inplace=True, errors='ignore')
        self.df = pd.concat([self.df, results], axis=1)
        self.ppnan(fillna_cols=new_cols.flatten(), fillna_value=-1)

    @staticmethod
    def apply_multiprocessing(*args):
        records, arange_labels, month_window, bucket_col, amount_col, date_col, key_col, row = args
        t = records[(records[date_col] >= row[2] - month_window) & (records[date_col] < row[2]) & (records[key_col] == row[3])]

        total_a = t[amount_col].mean()
        total_b = t[bucket_col].count()
        result = []

        for i in arange_labels:
            a = t[t[bucket_col] == i][amount_col].mean()
            b = t[t[bucket_col] == i][bucket_col].count()

            mean = (a / total_a) if total_a > 0 else -1
            count = (b / total_b) if total_b > 0 else -1
            result.extend([mean, count])

        return result
