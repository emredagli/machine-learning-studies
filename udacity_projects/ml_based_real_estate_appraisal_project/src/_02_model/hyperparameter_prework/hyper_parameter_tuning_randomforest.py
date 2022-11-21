# https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/

import sklearn
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np


# seed = np.random.seed(22)

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = RandomForestRegressor()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
# define search space
space = {
        'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

search = RandomizedSearchCV(model, space,
                            n_iter=60,
                            # scoring='neg_mean_absolute_error',
                            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
                            n_jobs=-1,
                            cv=cv,
                            random_state=1,
                            verbose=2)
result = search.fit(X, y)
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

