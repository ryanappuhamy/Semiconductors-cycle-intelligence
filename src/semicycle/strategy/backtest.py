"""Monthly backtest engine with explicit transaction costs.

Timing convention (no look-ahead):
  * at the close of month ``t`` we observe the signal and set target weight ``w_t``
  * ``w_t`` is held through month ``t+1``; the return earned in month ``t`` is
    ``w_{t-1} · r_t``
  * the trade into ``w_{t-1}`` (done at the close of ``t-1``) costs
    ``tc · |w_{t-1} - w_{t-2}|``, charged against month ``t``
"""

from __future__ import annotations

import itertools

import pandas as pd

from .signal import build_weights


def monthly_returns(prices: pd.Series) -> pd.Series:
    return prices.sort_index().pct_change().rename("ret")


def run_backtest(
    weights: pd.Series,
    asset_ret: pd.Series,
    *,
    cost_bps: float,
    benchmark_ret: pd.Series | None = None,
) -> pd.DataFrame:
    tc = cost_bps / 1e4
    # run only over the window the weights actually cover — never ffill a stale
    # weight forward onto later returns
    lo, hi = weights.index.min(), weights.index.max()
    idx = weights.index.union(asset_ret.index).sort_values()
    idx = idx[(idx >= lo) & (idx <= hi)]
    w = weights.reindex(idx).ffill()
    r = asset_ret.reindex(idx)

    w_held = w.shift(1)                      # weight in force during each month
    turnover = (w_held - w_held.shift(1)).abs()
    gross = w_held * r
    cost = tc * turnover
    net = gross - cost

    df = pd.DataFrame(
        {
            "weight": w_held,
            "asset_ret": r,
            "gross_ret": gross,
            "cost": cost,
            "strategy_ret": net,
            "turnover": turnover,
        }
    )
    if benchmark_ret is not None:
        df["benchmark_ret"] = benchmark_ret.reindex(idx)
    df = df.dropna(subset=["strategy_ret", "asset_ret"])

    df["strategy_cum"] = (1 + df["strategy_ret"]).cumprod()
    df["asset_cum"] = (1 + df["asset_ret"]).cumprod()
    if "benchmark_ret" in df:
        df["benchmark_cum"] = (1 + df["benchmark_ret"].fillna(0)).cumprod()
    return df


def grid_returns(panel: pd.DataFrame, asset_ret: pd.Series, base_cfg,
                 grid: dict[str, list]) -> pd.DataFrame:
    """A monthly strategy-return series for every config in the grid — the input
    to the deflated Sharpe ratio and the PBO."""
    keys = list(grid)
    cols = {}
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = base_cfg.model_copy(update=dict(zip(keys, combo, strict=True)))
        w = build_weights(panel, cfg)["weight"]
        bt = run_backtest(w, asset_ret, cost_bps=base_cfg.cost_bps)
        cols[", ".join(f"{k}={v}" for k, v in zip(keys, combo, strict=True))] = bt["strategy_ret"]
    return pd.DataFrame(cols).dropna(how="all")
