"""Model zoo for the nowcast: an autoregressive benchmark plus two learners.

Every model exposes the sklearn ``fit(X, y)`` / ``predict(X)`` interface and
tolerates missing values (linear models impute+scale in a pipeline; LightGBM
handles NaNs natively).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ColumnSelector(BaseEstimator):
    """Pass through only the named columns (kept for the AR benchmark)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame):  # noqa: N803
        cols = [c for c in self.columns if c in X.columns]
        return X[cols].to_numpy()


class ARBenchmark(BaseEstimator, RegressorMixin):
    """OLS on the point-in-time target and a few of its lags — an autoregression
    expressed only in terms genuinely available at decision time. This is the
    bar the feature-based models must clear."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self._pipe = Pipeline(
            [
                ("select", ColumnSelector(columns)),
                ("impute", SimpleImputer(strategy="median")),
                ("ols", LinearRegression()),
            ]
        )

    def fit(self, X, y):  # noqa: N803
        self._pipe.fit(X, y)
        return self

    def predict(self, X):  # noqa: N803
        return self._pipe.predict(X)


def _linear_pipe(l1_ratios, n_alphas) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    l1_ratio=list(l1_ratios),
                    alphas=int(n_alphas),  # sklearn>=1.7: int gives that many auto alphas
                    cv=4,
                    max_iter=10000,
                    n_jobs=-1,
                    random_state=0,
                ),
            ),
        ]
    )


class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = params
        self._model = None

    def fit(self, X, y):  # noqa: N803
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(**self.params, verbosity=-1)
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):  # noqa: N803
        return self._model.predict(np.asarray(X, dtype=float))


def make_models(cfg, benchmark_cols: list[str]) -> dict:
    m = cfg.params.models
    models = {
        "ar_benchmark": ARBenchmark(benchmark_cols),
        "elasticnet": _linear_pipe(
            m.elasticnet.get("l1_ratios", [0.5, 1.0]),
            m.elasticnet.get("n_alphas", 50),
        ),
        "lightgbm": LGBMWrapper(**m.lightgbm),
    }
    return models
