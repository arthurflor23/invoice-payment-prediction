from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import is_classifier, is_regressor, clone

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler
from yellowbrick.model_selection import rfecv

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

    def load(self, step, train=True, test=True):
        json_features = f'../assets/features{step}.json' \
            if os.path.exists(f'../assets/features{step}.json') else '../assets/features.json'

        self.x_columns = json.load(open(json_features))['x_columns']
        self.y_columns = ['PaidLate', 'PaidLateAM', 'DaysLate', 'DaysLateAM', 'PaymentCategory']

        df = pd.read_csv(self.filename, parse_dates=['DueDate'], low_memory=False)
        df['DaysLateAM'].clip(upper=30, inplace=True)

        t1 = df[df['DueDate'] < self.date]
        t2 = df[(df['DueDate'] >= self.date) & (df['DueDate'] < self.date + self.offset)]

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
            self.train = t1[t1['PaidLate'] == 1]
            self.test = t2[t2['PaidLate'] == 1]

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
        self.module = '.'.join(estimator.split('.')[:-1])
        self.estimator = estimator.split('.')[-1]
        self.model = getattr(importlib.import_module(self.module), self.estimator)

        self.searchfile = '../assets/gridsearch.json'
        self.hyperfile = f'../assets/hyperparameters_{step}.json'

        self.gridsearch = json.load(open(self.searchfile)) if os.path.exists(self.searchfile) else None
        self.hyper = json.load(open(self.hyperfile)) if os.path.exists(self.hyperfile) else {}
        self.output = os.path.join('..', 'output', f'step{step}', self.estimator)

    def selection(self, step, x_columns, x_train, y_train):
        vis = rfecv(self.model(random_state=SEED), X=x_train, y=y_train,
                    cv=StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=SEED))

        vis.show(outpath=f'../assets/features{step}.png')

        json_content = {'x_columns': np.array(x_columns)[vis.support_].tolist()}
        print(vis.support_, '\n', json_content['x_columns'])

        with open(f'../assets/features{step}.json', 'w') as f:
            json.dump(json_content, f, indent=4)

    def tuning(self, x_train, y_train):
        assert self.gridsearch
        param_grid = self.gridsearch[self.estimator] if self.estimator in self.gridsearch.keys() else {}

        if is_classifier(self.model):
            scoring = 'f1_macro'
            md = self.model(random_state=SEED)
            cv = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=SEED)

            if 'base_estimator' in md.get_params() and md.get_params()['base_estimator'] is None:
                base_param = {'base_estimator': DecisionTreeClassifier(class_weight='balanced', random_state=SEED)}
                md = md.set_params(**base_param)

        elif is_regressor(self.model):
            scoring = 'neg_mean_squared_error'
            md = self.model(random_state=SEED)
            cv = ShuffleSplit(n_splits=1, train_size=0.9, random_state=SEED)

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
        md = pickle.load(open(os.path.join(self.output, 'model.sav'), 'rb'))

        pd_test = md.predict(x_test.values)
        matches = self._report(y_test, pd_test, md, prefix='test')

        return y_test, pd_test, matches, self.output

    def train(self, x_train, y_train):
        md = self.model(random_state=SEED)

        if 'base_estimator' in md.get_params() and md.get_params()['base_estimator'] is None:
            if is_classifier(self.model):
                base_param = {'base_estimator': DecisionTreeClassifier(random_state=SEED)}

            elif is_regressor(self.model):
                base_param = {'base_estimator': DecisionTreeRegressor(random_state=SEED)}

            md = md.set_params(**base_param)

        if self.estimator in self.hyper.keys():
            md.set_params(**self.hyper[self.estimator]['params'])

        gt_train, pd_train, md = self._cross_validation(md, x_train, y_train, run_only_once=True)
        matches = self._report(gt_train, pd_train, md, prefix='train')

        return gt_train, pd_train, matches, self.output

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

    def _report(self, gt, pred, model, prefix):
        os.makedirs(self.output, exist_ok=True)
        pickle.dump(model, open(os.path.join(self.output, 'model.sav'), 'wb'))
        matches = None

        if is_classifier(self.model):
            with open(os.path.join(self.output, f'{prefix}_classification_report.txt'), 'w') as f:
                matches, report = [gt, pred], classification_report(gt, pred, digits=4, zero_division=True)
                f.write(''.join(report))
                print(''.join(report))

        elif is_regressor(self.model):
            with open(os.path.join(self.output, f'{prefix}_rmse_accuracy.txt'), 'w') as f:
                matches, report = self._acc_tolerance(gt, pred, tolerance=[0, 1, 2, 3])
                f.write('\n'.join(report))
                print('\n'.join(report))

        return matches

    def _acc_tolerance(self, gt, pred, tolerance=[0]):
        matches, report = [], [f'RMSE: {mean_squared_error(gt, pred, squared=False):.8}']

        for t in tolerance:
            match = []

            for i, x in enumerate(gt):
                p = np.array(pred[i], dtype=int)
                match.append(1 if p >= (x - t) and p <= (x + t) else 0)

            matches.append(np.array(match))
            acc = '{:.2f}%'.format(np.mean(match) * 100)
            report.append(f'Accuracy with ±{t} days of tolerance (range of {t*2} days): {acc}')

        return [tolerance, matches], report


def plot_classifier_report(y, pred, output, prefix, cmap='Blues'):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    g = sns.heatmap(confusion_matrix(y, pred), fmt='d', square=True, annot=True, cmap=cmap, ax=ax)
    g.get_figure().savefig(os.path.join(output, f'{prefix}_heatmap.png'))


def plot_classifier_per_day_report(matches, output, prefix, days):
    x_labels = 'Dia do Mês'
    y_labels = ['F1-score', 'Precision', 'Recall']
    p_labels = list(range(0, 101, 10))
    m_labels = list(range(0, 31, 2))

    df1 = pd.DataFrame({'Days': days, 'GT': matches[0], 'Predict': matches[1]})
    df2 = pd.DataFrame({x_labels: df1.groupby(['Days']).groups.keys(), y_labels[0]: 0, y_labels[1]: 0, y_labels[2]: 0})

    for i, d in enumerate(df1.groupby(['Days']).groups.keys()):
        gt_day = df1[df1['Days'] == d]['GT']
        pd_day = df1[df1['Days'] == d]['Predict']

        df2[y_labels[0]][i] = f1_score(gt_day, pd_day, average='macro', zero_division=True) * 100
        df2[y_labels[1]][i] = precision_score(gt_day, pd_day, zero_division=True) * 100
        df2[y_labels[2]][i] = recall_score(gt_day, pd_day, zero_division=True) * 100

    df2.plot.line(x=x_labels, y=y_labels, yticks=p_labels, xticks=m_labels, figsize=(8, 5), fontsize=14, rot=0)

    plt.ylabel('Score (%)', fontsize=14)
    plt.xlabel(x_labels, fontsize=14)
    plt.savefig(os.path.join(output, f'{prefix}_metrics_per_day.png'))


def plot_regressor_report(matches, output, prefix):
    x_label, y_label = 'Tolerância', 'Acurácia'
    x_labels = [f'±{x} dias' for x in matches[0]]
    p_labels = list(range(0, 101, 10))

    df1 = pd.DataFrame({x_label: x_labels, y_label: np.mean(matches[1], axis=1) * 100})
    ax = df1.plot.bar(x=x_label, y=y_label, yticks=p_labels, figsize=(8, 5), fontsize=14, rot=0, legend=False)

    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(f'{y_label} (%)', fontsize=14)

    for i, rect in enumerate(ax.patches):
        x = rect.get_x() + rect.get_width() / 2
        xy = rect.get_height() + 1
        ax.text(x, xy, '{:.2f}%'.format(df1[y_label][i]), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(output, f'{prefix}_accuracy.png'))


def plot_regressor_per_day_report(matches, output, prefix, days):
    x_labels = 'Dia do Mês'
    y_labels = [f'±{x} dias' for x in matches[0]]
    p_labels = list(range(0, 101, 10))
    m_labels = list(range(0, 31, 2))

    columns = {'Days': days}

    for i, x in enumerate(y_labels):
        columns[x] = matches[1][i]

    df1 = pd.DataFrame(columns)
    columns = {x_labels: df1.groupby(['Days']).groups.keys()}

    for i, x in enumerate(y_labels):
        columns[x] = df1.groupby(['Days'])[x].mean() * 100

    df2 = pd.DataFrame(columns)
    df2.plot.line(x=x_labels, y=y_labels, yticks=p_labels, xticks=m_labels, figsize=(8, 5), fontsize=14, rot=0)

    plt.ylabel('Acurácia (%)', fontsize=14)
    plt.xlabel(x_labels, fontsize=14)
    plt.savefig(os.path.join(output, f'{prefix}_accuracy_per_day.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int)
    parser.add_argument('-a', '--action', type=str)
    parser.add_argument('-e', '--estimator', type=str)
    arg = parser.parse_args()

    dataset = Dataset(filename=os.path.join('..', 'data', 'base4.csv'), split_date='2021-02-01')

    if arg.action == 'tuning':
        dataset.load(step=arg.step, train=True, test=False)
        model = Model(arg.estimator, step=arg.step)
        model.tuning(dataset.x_train, dataset.y_train)

    elif arg.action == 'selection':
        dataset.load(step=arg.step, train=True, test=False)
        model = Model(arg.estimator, step=arg.step)
        model.selection(arg.step, dataset.x_columns, dataset.x_train, dataset.y_train)

    else:
        if arg.action == 'train':
            dataset.load(step=arg.step, train=True, test=False)
            model = Model(arg.estimator, step=arg.step)
            y, p, m, o = model.train(dataset.x_train, dataset.y_train)

        elif arg.action == 'test':
            dataset.load(step=arg.step, train=False, test=True)
            model = Model(arg.estimator, step=arg.step)
            y, p, m, o = model.test(dataset.x_test, dataset.y_test)

        if arg.step == 1 or arg.step == 2:
            plot_classifier_report(y, p, o, prefix=arg.action, cmap=('Blues' if arg.step == 1 else 'YlOrBr'))

            if arg.action == 'test':
                plot_classifier_per_day_report(m, o, prefix='test', days=dataset.test['DueDate'].dt.day)

        elif arg.step == 3:
            plot_regressor_report(m, o, prefix=arg.action)

            if arg.action == 'test':
                plot_regressor_per_day_report(m, o, prefix='test', days=dataset.test['DueDate'].dt.day)
