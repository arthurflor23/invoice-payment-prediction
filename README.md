# Invoice Payment Prediction

Data mining project to predict late payment using decision tree algorithms. The problem was modeled in three steps:

1. Classification of payment of invoices between: on time and late;
2. Classification of late payment of invoices between: in the due month and later;
3. Estimated number of days overdue for overdue invoices.

## Installation

````
pip install -r requirements.tx
````

## Supported Estimators

### Classifier

* [sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [sklearn.ensemble.BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
* [imblearn.ensemble.BalancedBaggingClassifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html)
* [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [imblearn.ensemble.BalancedRandomForestClassifier](https://imbalancedlearn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html)
* [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [imblearn.ensemble.RUSBoostClassifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.html)
* [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)

### Regressor

* [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
* [sklearn.ensemble.BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)
* [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
* [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [xgboost.XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)

## Feature Selection

````
python model.py --step=<STEP_NUMPER> --action=selection --estimator=<ESTIMATOR_MODULE>
````

## Grid Search CV

````
python model.py --step=<STEP_NUMPER> --action=tuning --estimator=<ESTIMATOR_MODULE>
````

## Train

````
python model.py --step=<STEP_NUMPER> --action=train --estimator=<ESTIMATOR_MODULE>
````

## Test

````
python model.py --step=<STEP_NUMPER> --action=test --estimator=<ESTIMATOR_MODULE>
````