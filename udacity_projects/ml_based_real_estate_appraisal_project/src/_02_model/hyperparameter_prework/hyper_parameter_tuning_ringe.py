# https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/

# random search linear regression model on the auto insurance dataset
import sklearn
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import numpy as np


seed = np.random.seed(22)

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
dataframe = read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge()
# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]

# without hyperparameter:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test, y_pred))
# Mean Absolute Error: 26.44842022130946
# Mean Absolute Error: 30.575169928572492
# Mean Absolute Error: 22.659575876374916
# Mean Absolute Error: 25.920825033254175
# Mean Absolute Error: 28.676042384869177

# # define search
search = RandomizedSearchCV(model, space,
                            n_iter=100,
                            # scoring='neg_mean_absolute_error',
                            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
                            # n_jobs=-1,
                            cv=cv,
                            random_state=1)
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)


# Best Score: -29.228312319083592
# Best Hyperparameters: {'alpha': 0.00021225695021496544, 'fit_intercept': True, 'solver': 'sag'}

# Best Score: -29.13572046832377
# Best Hyperparameters: {'alpha': 0.00021225695021496544, 'fit_intercept': True, 'solver': 'sag'}

# Best Score: -29.184520411119422
# Best Hyperparameters: {'alpha': 0.0006169022534258897, 'fit_intercept': True, 'solver': 'sag'}

# Best Score: -28.94240577510493
# Best Hyperparameters: {'alpha': 0.26104441282754404, 'fit_intercept': True, 'solver': 'sag'}