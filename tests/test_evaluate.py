"""Walk-forward CV must purge and embargo — no training row may sit inside the
forecast window of the test month."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from semicycle.nowcast.dataset import Supervised
from semicycle.nowcast.evaluate import walk_forward


class SpyModel(BaseEstimator, RegressorMixin):
    """Records the last training index it was fitted on."""

    seen: list = []

    def fit(self, X, y):  # noqa: N803
        SpyModel.seen.append(X.index)
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):  # noqa: N803
        return np.full(len(X), self._mean)


def _make_data(horizon: int) -> Supervised:
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    X = pd.DataFrame({"f": np.random.default_rng(0).normal(size=240)}, index=idx)
    y = pd.Series(np.arange(240, dtype=float), index=idx)
    return Supervised(X=X, y=y, horizon=horizon, feature_names=["f"], benchmark_cols=[])


def test_purge_and_embargo_gap():
    SpyModel.seen.clear()
    data = _make_data(horizon=3)
    purge, embargo = 3, 1
    gap = data.horizon + purge + embargo  # 7 months

    results = walk_forward(
        data, {"spy": SpyModel()},
        min_train_months=60, step_months=6,
        purge_months=purge, embargo_months=embargo,
    )
    assert len(results) > 0

    for test_date, train_idx in zip(results.index, SpyModel.seen, strict=False):
        months = (test_date.year - train_idx.year) * 12 + (test_date.month - train_idx.month)
        # every training row is at least `gap` months before the test month
        assert months.min() >= gap


def test_no_training_row_after_test_month():
    SpyModel.seen.clear()
    data = _make_data(horizon=0)
    walk_forward(
        data, {"spy": SpyModel()},
        min_train_months=60, step_months=12,
        purge_months=2, embargo_months=1,
    )
    for test_date, train_idx in zip(
        walk_forward(
            data, {"spy": SpyModel()},
            min_train_months=60, step_months=12, purge_months=2, embargo_months=1,
        ).index,
        SpyModel.seen[-100:],
        strict=False,
    ):
        assert train_idx.max() < test_date
