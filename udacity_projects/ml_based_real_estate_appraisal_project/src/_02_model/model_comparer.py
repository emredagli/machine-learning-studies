from typing import List, Dict, Final
from sklearn import model_selection
import matplotlib.pyplot as plt
from math import floor, ceil
import pandas
import numpy as np

from _02_model.regression_model import RegressionModel

from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV


class ModelComparer:
    _models: List[RegressionModel]
    _scoring: Dict[str, str]
    _cross_val_results: Dict[str, List[List[float]]]
    _cross_val_models: List[str]
    _random_seed: int = None
    _X = None
    _y = None
    _n_splits = 2

    SCORING_PREFIX: Final = 'test_'

    def __init__(self, x, y, models, scoring, random_seed=None):
        self._X = x
        self._y = y
        self._models = models
        self._scoring = scoring
        self._random_seed = random_seed
        self._set_seed()

    def init_models(self):
        for model in self._models:
            model.init_model()

    def cross_validate(self):
        self._init_cross_val_results()
        for model in self._models:
            self._cross_val_models.append(model.name)
            k_fold = model_selection.KFold(n_splits=self._n_splits, random_state=self._random_seed, shuffle=True)
            cross_val_results = model_selection.cross_validate(model.instance, self._X, self._y, cv=k_fold,
                                                               scoring=self._scoring)
            for metric_key in self._cross_val_results.keys():
                self._cross_val_results.get(metric_key).append(cross_val_results.get(metric_key))

    def _init_cross_val_results(self):
        self._cross_val_models = []
        self._cross_val_results = {
            "fit_time": [],
            "score_time": []
        }

        for metric_key in self._scoring.keys():
            self._cross_val_results[self.SCORING_PREFIX + metric_key] = []

    def optimize_hyperparameters_with_random_search(self, scoring):
        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)

        for model in self._models:
            if model.hyper_space is not None:
                self._set_seed()
                search = RandomizedSearchCV(model.get_default_instance(), model.hyper_space,
                                            n_iter=100,
                                            scoring=scoring,
                                            n_jobs=self._get_n_jobs(),
                                            cv=cv,
                                            random_state=1,
                                            verbose=2)

                result = search.fit(self._X, self._y)

                # Ridge stored results
                # result = {
                #     "best_score_": -1.5349370836212146,
                #     "best_params_": {"alpha": 1.125207932754771e-05, "fit_intercept": True, "solver": "svd"}
                # }

                print(model.name, {
                    "best_score_": result["best_score_"],
                    "best_params_": result["best_params_"]
                })
                model.set_hyperparameters(result)

    def plot_results(self):

        metric_keys = self._cross_val_results.keys()
        nc = 3
        nr = ceil(len(metric_keys) / nc)
        inches = 4
        fig = plt.figure(figsize=(nc * inches * 1.4, nr * inches))
        fig.suptitle(f'Algorithm Comparison\nWith {self._n_splits}-fold Crossvalidation')
        plt.tight_layout()

        for index, metric_key in enumerate(metric_keys):
            ax = fig.add_subplot(nr, nc, index + 1)
            plt.boxplot(self._cross_val_results[metric_key])
            ax.title.set_text(metric_key.replace(self.SCORING_PREFIX, ''))
            ax.set_xticklabels(self._cross_val_models, rotation=-20, ha='left')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid()

        plt.show()

    def _set_seed(self):
        if self._random_seed is not None:
            np.random.seed(22)

    def _get_n_jobs(self):
        """
        To get the same results to properly test and compare the results, sometimes we need to use fixed seed.
        But n_jobs field should be None (means 1) to use same seed to keep same randomness over the experiments.
        """
        if self._random_seed is not None:
            return None
        else:
            return -1
