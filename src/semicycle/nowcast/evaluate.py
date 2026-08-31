"""Expanding-window walk-forward evaluation with purge and embargo.

For each out-of-sample month ``t`` we retrain on everything old enough that its
target window cannot overlap ``t``'s:

    train rows d  with  d + horizon + purge + embargo  <=  t   (in months)

``purge`` removes rows whose label interval ``[d, d+horizon]`` reaches into the
test month (López de Prado); ``embargo`` adds a further gap so slow-moving
autocorrelation cannot leak across the boundary. Each fold is a genuine
out-of-sample forecast made with information available at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .dataset import Supervised
from .models import ARBenchmark


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def walk_forward(
    data: Supervised,
    models: dict,
    *,
    min_train_months: int,
    step_months: int = 1,
    purge_months: int = 3,
    embargo_months: int = 1,
    oos_start: str | None = None,
) -> pd.DataFrame:
    idx = data.X.index
    horizon = data.horizon
    gap = horizon + purge_months + embargo_months

    first_test_pos = min_train_months + gap
    if oos_start is not None:
        start_ts = pd.Timestamp(oos_start)
        first_test_pos = max(first_test_pos, int((idx < start_ts).sum()))
    test_positions = range(first_test_pos, len(idx), step_months)

    rows = []
    for pos in test_positions:
        t = idx[pos]
        train_mask = np.array([_months_between(d, t) >= gap for d in idx])
        train_mask &= np.arange(len(idx)) < pos
        if train_mask.sum() < min_train_months:
            continue

        X_tr, y_tr = data.X.loc[train_mask], data.y.loc[train_mask]
        X_te = data.X.iloc[[pos]]

        rec = {"date": t, "y_true": float(data.y.iloc[pos]), "n_train": int(train_mask.sum())}
        for name, model in models.items():
            est = ARBenchmark(model.columns) if isinstance(model, ARBenchmark) else clone(model)
            est.fit(X_tr, y_tr)
            rec[f"pred_{name}"] = float(np.asarray(est.predict(X_te))[0])
        rows.append(rec)

    return pd.DataFrame(rows).set_index("date")


def _directional_hit(y_true: pd.Series, pred: pd.Series) -> float:
    """Share of months where the model gets the sign of the cycle right
    (accelerating vs contracting YoY growth)."""
    mask = y_true.notna() & pred.notna()
    return float((np.sign(y_true[mask]) == np.sign(pred[mask])).mean())


def _turn_hit(y_true: pd.Series, pred: pd.Series, restrict: pd.Series | None = None) -> float:
    """Share of months where the model calls the direction of change correctly:
    does ``pred_t`` sit on the right side of ``y_{t-1}`` relative to ``y_t``?"""
    prev = y_true.shift(1)
    dt = np.sign(y_true - prev)
    dp = np.sign(pred - prev)
    mask = prev.notna() & pred.notna()
    if restrict is not None:
        mask &= restrict.reindex(mask.index).fillna(False)
    return float((dt[mask] == dp[mask]).mean())


def scoreboard(
    results: pd.DataFrame, *, since: str | None = None, turn_quantile: float = 0.67
) -> pd.DataFrame:
    """Model comparison. Alongside the full-sample error, `MAE_turns` /
    `turn_acc_turns` restrict to the months where the cycle is actually moving
    (|Δ realised YoY| above its `turn_quantile`) — the inflections where a cycle
    signal has to earn its keep, and where the near-random-walk AR benchmark is
    weakest."""
    if since is not None:
        results = results.loc[results.index >= pd.Timestamp(since)]
    y = results["y_true"]
    pred_cols = [c for c in results.columns if c.startswith("pred_")]
    bench = "pred_ar_benchmark"
    bench_mae = (y - results[bench]).abs().mean() if bench in results else np.nan

    dy = y.diff().abs()
    turn_mask = dy >= dy.quantile(turn_quantile)

    rows = []
    for c in pred_cols:
        err = y - results[c]
        mae = err.abs().mean()
        turn_mae = err[turn_mask].abs().mean()
        rows.append(
            {
                "model": c.removeprefix("pred_"),
                "MAE": mae,
                "RMSE": float(np.sqrt((err**2).mean())),
                "skill_vs_AR": np.nan if np.isnan(bench_mae) else 1 - mae / bench_mae,
                "dir_acc": _directional_hit(y, results[c]),
                "turn_acc": _turn_hit(y, results[c]),
                "corr": float(results[c].corr(y)),
                "MAE_turns": turn_mae,
                "turn_acc_turns": _turn_hit(y, results[c], restrict=turn_mask),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("MAE")
