import datetime
import functools
import multiprocessing
import numpy as np
import pandas as pd


class CSVManager():
    def __init__(self):
        self.df = None
        self.bins = None
        self.labels = None

    def read(self, filename, sep=';', na_values=['N/I'], orderby=None, ascending=True):
        self.df = pd.read_csv(filename, sep=sep, na_values=na_values)

        if orderby is not None:
            self.df.sort_values(by=orderby, ascending=ascending, ignore_index=True, inplace=True)

    def save(self, name, sep=';', na_rep=-1):
        self.df.to_csv(name, sep=sep, na_rep=na_rep, float_format='%g', index=False)

    def info(self):
        print(self.df.info())

    def head(self, value=10):
        print(self.df.head(value))

    def null_sum(self):
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

    def minmax_filter(self, min_cols, max_cols):
        for mi, ma in zip(min_cols, max_cols):
            self.df = self.df[self.df[mi] < self.df[ma]]

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

    def calculate_per_bucket(self, bucket_col, amount_col, date_col, key_col, month_window):
        self.df.reset_index(drop=True, inplace=True)
        dflocal = self.df[[bucket_col, amount_col, date_col, key_col]].copy()

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

        new_cols = np.array([[f'Range{i}Amount', f'Range{i}Count'] for i in arange_labels])
        results = pd.DataFrame(np.array(results), columns=new_cols.flatten())

        self.df.drop(labels=new_cols.flatten(), axis=1, inplace=True, errors='ignore')
        self.df = pd.concat([self.df, results], axis=1)

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


if __name__ == '__main__':
    csv = CSVManager()
    csv.read('InvoicedDocuments_v4.csv', orderby=['DocumentDate'])
    csv.ppnan(dropna_cols=['ClearingDate'], fillna_value=0)

    csv.cast_to_number(cols=['CustomerRegion', 'PaymentTerms'])
    csv.cast_to_integer(cols=['InvoicedDocuments', 'PaidDocuments', 'PaidPastDocuments', 'OpenDocuments', 'PastDueDocuments'])
    csv.cast_to_date(cols=['CustomerLastCreditReview', 'DocumentDate', 'DueDate', 'ClearingDate'])

    csv.minmax_filter(min_cols=['DocumentDate', 'DocumentDate'], max_cols=['DueDate', 'ClearingDate'])
    csv.extract_days(cols=['DocumentDate', 'DueDate'])

    csv.set_ratio(dividend_cols=['InvoicedAmount', 'PaidAmount', 'PaidPastAmount', 'OpenAmount', 'PastDueAmount'],
                  divisor_cols=['InvoicedDocuments', 'PaidDocuments', 'PaidPastDocuments', 'OpenDocuments', 'PastDueDocuments'])

    csv.set_daysto(source_cols=['DocumentDate', 'DocumentDate', 'CustomerLastCreditReview'],
                   target_cols=['DueDate', 'ClearingDate', 'DocumentDate'])

    csv.set_range(bins=list(range(1, 31, 28)), cols=['DaysToDueDate', 'DaysToClearingDate'])

    csv.calculate_per_bucket(bucket_col='DaysToClearingDateRangeCT', amount_col='DocumentAmount',
                             date_col='DocumentDate', key_col='CustomerKey', month_window=2)

    # train, test = csv.get_data_range(col='DocumentDate', date='2020-08-01', month_window=4)

    # csv.info()
    # csv.head(20)
    csv.save(name='InvoicedDocuments_v4_pp.csv')
