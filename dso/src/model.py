from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import is_classifier, is_regressor, clone
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import importlib
import argparse
import pickle
import json
import os

SEED = 42


class Dataset():
    def __init__(self, filename, split_date):
        self.filename = filename
        self.date = pd.to_datetime(split_date)
        self.offset = pd.DateOffset(months=1)

    def set_columns(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def load(self, step, train=True, test=True):
        df = pd.read_csv(self.filename, parse_dates=['DueDate'], low_memory=False)

        t1 = df[df['DueDate'] < self.date].drop(['DueDate'], axis=1)
        t2 = df[(df['DueDate'] >= self.date) & (df['DueDate'] < self.date + self.offset)].drop(['DueDate'], axis=1)

        if step == 1:
            self.y = 'PaidLate'
            self.train = t1
            self.test = t2
        elif step == 2:
            self.y = 'PaidLateAM'
            self.train = t1[t1['PaidLate'] == 1]
            self.test = t2[t2['PaidLate'] == 1]
        elif step == 3:
            self.y = 'DaysLateAM'
            self.train = t1[t1['PaidLateAM'] == 1]
            self.test = t2[t2['PaidLateAM'] == 1]

        if self.x_columns and self.y_columns:
            self.transform(self.y, train, test)

    def transform(self, y, train=True, test=True):
        if train:
            self.x_train = self.train[self.x_columns]
            self.y_train = self.train[self.y_columns][y]

            self.qt_train = RobustScaler(quantile_range=(25.0, 75.0))
            self.x_train.iloc[:][:] = self.qt_train.fit_transform(self.x_train.to_numpy())

        if test:
            self.x_test = self.test[self.x_columns]
            self.y_test = self.test[self.y_columns][y]

            self.qt_test = RobustScaler(quantile_range=(25.0, 75.0))
            self.x_test.iloc[:][:] = self.qt_test.fit_transform(self.x_test.to_numpy())


class Model():
    def __init__(self, estimator, step):
        self.module = '.'.join(estimator.split('.')[:2])
        self.estimator = '.'.join(estimator.split('.')[2:])
        self.model = getattr(importlib.import_module(self.module), self.estimator)

        self.searchfile = 'gridsearch.json'
        self.hyperfile = f'hyperparameters_{step}.json'

        self.gridsearch = json.load(open(self.searchfile)) if os.path.exists(self.searchfile) else None
        self.hyper = json.load(open(self.hyperfile)) if os.path.exists(self.hyperfile) else {}

        self.cmap = 'Blues' if step == 1 else 'YlOrBr'
        self.output = os.path.join('..', f'output{step}', self.estimator)

    def tunning(self, x_train, y_train):
        assert self.gridsearch
        param_grid = self.gridsearch[self.estimator] if self.estimator in self.gridsearch.keys() else {}

        if is_classifier(self.model):
            scoring = 'f1_macro'
            md = self.model(random_state=SEED)
            cv = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)

            if 'base_estimator' in md.get_params() and md.get_params()['base_estimator'] is None:
                base_param = {'base_estimator': DecisionTreeClassifier(class_weight='balanced', random_state=SEED)}
                md = md.set_params(**base_param)

        elif is_regressor(self.model):
            scoring = 'neg_mean_squared_error'
            md = self.model(random_state=SEED)
            cv = ShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)

            if 'base_estimator' in md.get_params() and md.get_params()['base_estimator'] is None:
                base_param = {'base_estimator': DecisionTreeRegressor(random_state=SEED)}
                md = md.set_params(**base_param)

        grid = GridSearchCV(md, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=3)
        grid.fit(x_train, y_train)

        print('Best score:', grid.best_score_)
        print('Best params:', grid.best_params_)

        self.hyper[self.estimator] = {'score': grid.best_score_, 'params': grid.best_params_}

        with open(self.hyperfile, 'w') as f:
            json.dump(self.hyper, f, indent=4)

    def test(self, x_test, y_test):
        model = pickle.load(open(os.path.join(self.output, 'model.sav'), 'rb'))

        pd_test = model.predict(x_test)
        self._report(y_test, pd_test, prefix='test', cmap=self.cmap)

    def train(self, x_train, y_train):
        md = self.model(random_state=SEED)

        if 'base_estimator' in md.get_params() and md.get_params()['base_estimator'] is None:
            if is_classifier(self.model):
                base_param = {'base_estimator': DecisionTreeClassifier(random_state=SEED)}
            elif is_regressor(self.model):
                base_param = {'base_estimator': DecisionTreeRegressor(random_state=SEED)}

        if self.estimator in self.hyper.keys():
            md = md.set_params(**base_param)

        md.set_params(**self.hyper[self.estimator]['params'])

        gt_train, pd_train, model = self._cross_validation(md, x_train, y_train, run_only_once=True)
        self._report(gt_train, pd_train, model, prefix='train', cmap=self.cmap)

    def _cross_validation(self, model, x, y, n_splits=10, n_repeats=3, run_only_once=False):
        if is_classifier(model):
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

        elif is_regressor(model):
            rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

        gt, pred = [], []
        x, y = x.to_numpy(), y.to_numpy()

        for i, (train, test) in enumerate(rskf.split(x, y)):
            if i == 0:
                print(f'Train size: {len(train)}, Test size: {len(test)}')

            print(f'Running: {i + 1} / {1 if run_only_once else rskf.get_n_splits()}')
            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]

            model_f = clone(model)
            model_f.fit(x_train, np.squeeze(y_train))

            gt.extend(y_test)
            pred.extend(model_f.predict(x_test))

            if run_only_once:
                break

        return gt, pred, model_f

    def _report(self, gt, pred, model, prefix, cmap='Blues'):
        os.makedirs(self.output, exist_ok=True)
        pickle.dump(model, open(os.path.join(self.output, 'model.sav'), 'wb'))

        if is_classifier(self.model):
            with open(os.path.join(self.output, f'{prefix}_classification_report.txt'), 'w') as f:
                report = classification_report(gt, pred, digits=4, zero_division=True)
                f.write(''.join(report))
                print(report)

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            g = sns.heatmap(confusion_matrix(gt, pred), fmt='d', square=True, annot=True, cmap=cmap, ax=ax)
            g.get_figure().savefig(os.path.join(self.output, f'{prefix}_heatmap.png'))

        elif is_regressor(self.model):
            with open(os.path.join(self.output, f'{prefix}_rmse_accuracy.txt'), 'w') as f:
                report = self._accuracy_tolerance(gt, pred, tolerance=[0, 1, 2, 3])
                f.write('\n'.join(report))
                print('\n'.join(report))

    def _accuracy_tolerance(self, gt, pd, tolerance=[0]):
        rmse = np.sqrt(mean_squared_error(gt, pd))
        report = [f'RMSE: {rmse:.8}']

        for t in tolerance:
            match = 0

            for i, x in enumerate(gt):
                match += 1 if pd[i] >= (x - t) and pd[i] <= (x + t) else 0

            acc = ((match / len(gt)) * 100)
            report.append(f'Accuracy with ±{t} days of tolerance (range of {t*2} days): {acc:.8}%')

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int)
    parser.add_argument('-a', '--action', type=str)
    parser.add_argument('-e', '--estimator', type=str)

    arg = parser.parse_args()

    dataset = Dataset(filename=os.path.join('..', 'data', 'base4.csv'),
                      split_date='2021-02-01')

    dataset.set_columns(x_columns=['InvoiceCount',
                                   'OSInvoiceCount',
                                   'R_OSInvoiceCount',

                                   'InvoiceAmount',
                                   'OSInvoiceAmount',
                                   'R_OSInvoiceAmount',

                                   'DaysToDueDate',
                                   'DaysToEndMonth',
                                   'WeekdayEndMonth',
                                   'PartnerCustomer',

                                   'MAD_DaysLate',
                                   'MED_DaysLate',

                                   'MAD_DaysLateAM',
                                   'MED_DaysLateAM',

                                   'MAD_OSDaysLate',
                                   'MED_OSDaysLate',

                                   'MAD_OSDaysLateAM',
                                   'MED_OSDaysLateAM',

                                   'PaidCount',
                                   'PaidLateCount',
                                   'PaidLateAMCount',
                                   'R_PaidLateCount',
                                   'R_PaidLateAMCount',

                                   'PaidAmount',
                                   'PaidLateAmount',
                                   'PaidLateAMAmount',
                                   'R_PaidLateAmount',
                                   'R_PaidLateAMAmount',

                                   'OSCount',
                                   'OSLateCount',
                                   'OSLateAMCount',
                                   'R_OSLateCount',
                                   'R_OSLateAMCount',

                                   'OSAmount',
                                   'OSLateAmount',
                                   'OSLateAMAmount',
                                   'R_OSLateAmount',
                                   'R_OSLateAMAmount'],

                        y_columns=['PaidLate',
                                   'PaidLateAM',
                                   'DaysLate',
                                   'DaysLateAM',
                                   'PaymentCategory'])

    if arg.action == 'tunning':
        dataset.load(step=arg.step, train=True, test=False)

        model = Model(arg.estimator, step=arg.step)
        model.tunning(dataset.x_train, dataset.y_train)

    elif arg.action == 'train':
        dataset.load(step=arg.step, train=True, test=False)

        model = Model(arg.estimator, step=arg.step)
        model.train(dataset.x_train, dataset.y_train)

    elif arg.action == 'test':
        dataset.load(step=arg.step, train=False, test=True)

        model = Model(arg.estimator, step=arg.step)
        model.test(dataset.x_test, dataset.y_test)
