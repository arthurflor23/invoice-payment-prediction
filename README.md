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

## Citation

If this project helped in any way in your research work, feel free to cite the following paper.

### Predição de Pagamentos Atrasados Através de Algoritmos Baseados em Árvore de Decisão ([here](http://revistas.poli.br/index.php/repa/article/view/1746))

```
@article{10.25286/repa.v6i5.1746,
    author    = {Neto, Arthur F. S. and Silva, José F. G. da and Oliveira, Glauber N. de},
    title     = {Predição de Pagamentos Atrasados Através de Algoritmos Baseados em Árvore de Decisão},
    journal   = {Revista de Engenharia e Pesquisa Aplicada (REPA)},
    pages     = {1-10},
    month     = {11},
    year      = {2021},
    volume    = {6},
    number    = {5},
    url       = {https://doi.org/10.25286/repa.v6i5.1746},
    doi       = {10.25286/repa.v6i5.1746},
}
```
