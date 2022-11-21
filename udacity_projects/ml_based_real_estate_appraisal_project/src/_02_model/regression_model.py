class RegressionModel:
    _model_class = None
    _model = None
    _hyper_space = None
    _hyperparameters_result = None
    _best_params = {}

    def __init__(self, model_class, hyper_space=None):
        self._model_class = model_class
        self._hyper_space = hyper_space

    def init_model(self):
        self._model = self._model_class(**self._best_params)

    @property
    def name(self):
        return self._model_class.__name__

    @property
    def instance(self):
        return self._model

    @property
    def hyper_space(self):
        return self._hyper_space

    def set_hyperparameters(self, result):
        self._hyperparameters_result = result
        self._best_params = result.get("best_params_", {})
        # print('Best Score: %s' % result.best_score_)
        # print('Best Hyperparameters: %s' % result.best_params_)

    def get_default_instance(self):
        return self._model_class()
