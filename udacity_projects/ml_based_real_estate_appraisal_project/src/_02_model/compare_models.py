from _02_model.model_comparer import ModelComparer
from _02_model.regression_model import RegressionModel
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
import numpy as np
import math
from scipy.stats import loguniform

# Linear Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression

# Lasso Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
from sklearn.linear_model import Lasso

# Ridge Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
from sklearn.linear_model import Ridge

# Random Forest Regressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor

# XGBoost Python API Referencehttps://xgboost.readthedocs.io/en/stable/python/python_api.html
# XGBoost parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
# XGBoost for Regression: https://machinelearningmastery.com/xgboost-for-regression/
from xgboost import XGBRegressor

# LightGBM: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
from lightgbm import LGBMRegressor

# Support Vector Regression: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
from sklearn.svm import SVR

# CatBoost:  https://catboost.ai/en/docs/concepts/python-installation
# from catboost import Pool, CatBoostRegressor  -> Pip install problem with Mac M1.

# Principal Components Regression (PCR)
# https://www.statology.org/principal-components-regression-in-python/

# Polynomial Regression
# PolynomialFeatures: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
# Polynomial Regression in Python using scikit-learn: https://data36.com/polynomial-regression-python-scikit-learn/


# Hyperparameters for models:
#  * https://pythonguides.com/scikit-learn-hyperparameter-tuning/

models = [
    RegressionModel(LinearRegression),
    # RegressionModel(Lasso),
    RegressionModel(Ridge),
    # RegressionModel(Ridge, hyper_space={
    #     "solver": ['svd', 'cholesky', 'lsqr', 'sag'],
    #     "alpha": loguniform(1e-5, 100),
    #     "fit_intercept": [True, False]
    # }),
    # RegressionModel(XGBRegressor),
    # RegressionModel(LGBMRegressor),
    RegressionModel(RandomForestRegressor, hyper_space={
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }),
    RegressionModel(RandomForestRegressor)
    # RegressionModel(SVR)
]

df = pd.read_csv("../../data/02_model/kc_house_data_cleaned_01.csv")

y = df.iloc[:, 0]
X = df.iloc[:, 1:]


# Predefined metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Regression Metrics:
# ‘explained_variance’
# ‘max_error’
# ‘neg_mean_absolute_error’
# ‘neg_mean_squared_error’
# ‘neg_root_mean_squared_error’
# ‘neg_mean_squared_log_error’
# ‘neg_median_absolute_error’
# ‘r2’
# ‘neg_mean_poisson_deviance’
# ‘neg_mean_gamma_deviance’
# ‘neg_mean_absolute_percentage_error’


def custom_root_mean_squared_error(y_actual, y_predicted):
    mse = np.square(np.subtract(y_predicted, y_actual)).mean()
    root_mse = math.sqrt(mse)
    return root_mse


def custom_mean_absolute_percentage_error(y_actual, y_predicted):
    mape = np.abs(np.divide(np.subtract(y_predicted, y_actual), y_actual)).mean()
    return mape * 100


scoring = {
    # 'neg_mean_squared_error': 'neg_mean_squared_error',
    # 'neg_mean_absolute_error': 'neg_mean_absolute_error',
    # 'neg_mean_squared_log_error': 'neg_mean_squared_log_error',
    'MSE': make_scorer(mean_squared_error),
    'RMSE': make_scorer(custom_root_mean_squared_error),  # , greater_is_better=False
    'MAPE': make_scorer(custom_mean_absolute_percentage_error)
}

model_comparer = ModelComparer(x=X, y=y, models=models, scoring=scoring, random_seed=None)

model_comparer.optimize_hyperparameters_with_random_search(
    scoring=make_scorer(custom_mean_absolute_percentage_error, greater_is_better=False)
)
model_comparer.init_models()
model_comparer.cross_validate()
model_comparer.plot_results()
