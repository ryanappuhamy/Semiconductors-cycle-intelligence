"""Backtest mechanics and the overfitting statistics."""

import numpy as np
import pandas as pd
import pytest

from semicycle.strategy.backtest import monthly_returns, run_backtest
from semicycle.strategy.stats import (
    deflated_sharpe_ratio,
    perf_stats,
    probability_of_backtest_overfitting,
)


@pytest.fixture
def rets():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.01, 0.05, 120), index=idx)


def test_backtest_no_lookahead(rets):
    # weight is 1.0 for one month only, at t=50
    w = pd.Series(0.0, index=rets.index)
    w.iloc[50] = 1.0
    d50, d51 = rets.index[50], rets.index[51]
    bt = run_backtest(w, rets, cost_bps=0)
    # the weight set at the close of month 50 earns the return of month 51, not 50
    assert bt.loc[d50, "strategy_ret"] == pytest.approx(0.0, abs=1e-12)
    assert bt.loc[d51, "strategy_ret"] == pytest.approx(rets.iloc[51])


def test_backtest_costs_reduce_return(rets):
    w = pd.Series(np.where(np.arange(120) % 2, 1.0, 0.0), index=rets.index)
    free = run_backtest(w, rets, cost_bps=0)["strategy_ret"].sum()
    costly = run_backtest(w, rets, cost_bps=50)["strategy_ret"].sum()
    assert costly < free
    # each flip is a full unit of turnover -> 50bps drag per month traded
    bt = run_backtest(w, rets, cost_bps=50)
    assert bt["cost"][bt["turnover"] > 0].round(4).eq(0.005).all()


def test_backtest_never_extends_past_weights(rets):
    w = pd.Series(1.0, index=rets.index[:60])
    bt = run_backtest(w, rets, cost_bps=0)
    assert bt.index.max() <= rets.index[59]


def test_monthly_returns():
    p = pd.Series([100, 110, 99], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    r = monthly_returns(p)
    assert r.iloc[1] == pytest.approx(0.1)
    assert r.iloc[2] == pytest.approx(-0.1)


def test_deflated_sharpe_shrinks_with_more_trials(rets):
    d1 = deflated_sharpe_ratio(rets, n_trials=1)
    d50 = deflated_sharpe_ratio(rets, n_trials=50)
    assert d50["expected_max_sharpe_H0"] > d1["expected_max_sharpe_H0"]
    assert 0.0 <= d50["deflated_sharpe_ratio"] <= 1.0


def test_pbo_identical_configs_is_half():
    idx = pd.date_range("2000-01-31", periods=160, freq="ME")
    rng = np.random.default_rng(3)
    base = rng.normal(0.008, 0.04, 160)
    m = pd.DataFrame({f"c{i}": base + rng.normal(0, 1e-6, 160) for i in range(8)}, index=idx)
    pbo = probability_of_backtest_overfitting(m, n_partitions=8)
    assert 0.3 <= pbo["pbo"] <= 0.7  # indistinguishable configs -> coin flip


def test_pbo_one_dominant_config_is_low():
    idx = pd.date_range("2000-01-31", periods=160, freq="ME")
    rng = np.random.default_rng(4)
    m = pd.DataFrame({f"c{i}": rng.normal(0.0, 0.04, 160) for i in range(7)}, index=idx)
    m["winner"] = rng.normal(0.02, 0.02, 160)  # genuinely better in every subsample
    pbo = probability_of_backtest_overfitting(m, n_partitions=8)
    assert pbo["pbo"] < 0.2


def test_perf_stats_keys(rets):
    st = perf_stats(rets)
    for k in ("sharpe", "ann_vol", "max_drawdown", "calmar", "skew"):
        assert k in st.index
    assert st["max_drawdown"] <= 0
