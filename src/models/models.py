import joblib
import lightgbm as lgb
import numpy as np
import statsmodels.api as sm
import xgboost as xgb

from abc import ABC, abstractmethod
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit


class ModelAbstractClass(ABC):
    def __init__(self, path=None):
        self.model = None
        self.__fitted = False

        if path is not None:
            self.load(path)

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def _predict(self, X):
        pass

    def fit(self, X, y):
        self._fit(X, y)
        self.__fitted = True

    def predict(self, X):
        assert self.__fitted is True
        return self._predict(X)

    def save(self, path):
        assert self.__fitted is True
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.__fitted = True


class StatsModelsLogisticRegression(ModelAbstractClass):
    def __init__(self, path=None):
        super().__init__(path=path)

    def _fit(self, X, y):
        y_count = np.bincount(y)

        y_w01 = y_count[0] / y_count[1]
        y_w01_weights = [1 + k * (y_w01 - 1) for k in y]  # Upweight if class 1

        X = sm.add_constant(X)

        model = GLM(y, X, freq_weights=y_w01_weights, family=Binomial(link=logit()))
        self.model = model.fit()

    def _predict(self, X):
        X = sm.add_constant(X)
        res = self.model.predict(X)

        return res


class XGBoostClassifier(ModelAbstractClass):
    def __init__(self, path=None, n_estimators=100, max_depth=3):
        super().__init__(path=path)

        if self.model is None:
            self.model = xgb.XGBClassifier(use_label_encoder=False,
                                           eval_metric='logloss',
                                           n_estimators=n_estimators,
                                           max_depth=max_depth)
            self.__fitted = False

    def _fit(self, X, y):
        self.model.fit(X, y)
        self.__fitted = True

    def _predict(self, X):
        res = self.model.predict_proba(X)[:, 1]

        return res


class LightGBMClassifier(ModelAbstractClass):
    def __init__(self, path=None, boosting_type='gbdt', n_estimators=100):
        super().__init__(path=path)
        if self.model is None:
            self.model = lgb.LGBMClassifier(boosting_type=boosting_type,
                                            n_estimators=n_estimators)
            self.__fitted = False

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        res = self.model.predict_proba(X)[:, 1]

        return res
