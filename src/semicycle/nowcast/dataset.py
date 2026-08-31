"""Turn the monthly panel into a supervised (X, y) frame for one horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Supervised:
    X: pd.DataFrame          # features, indexed by decision month T
    y: pd.Series             # target realised at T + horizon
    horizon: int
    feature_names: list[str]
    benchmark_cols: list[str]  # subset of X used by the AR benchmark


def make_supervised(panel: pd.DataFrame, horizon: int, *, min_coverage: float = 0.10) -> Supervised:
    """``y[T]`` = ``panel['target'][T + horizon]``; ``X[T]`` = all point-in-time
    features at ``T``. Feature columns that are mostly empty (e.g. a source that
    failed to download) are dropped."""
    if "target" not in panel.columns:
        raise KeyError("panel has no 'target' column")

    y = panel["target"].shift(-horizon).rename("y")

    drop = {"target", "target_pit"} | {c for c in panel.columns if c.startswith("target_pit__")}
    feat_cols = [c for c in panel.columns if c not in drop]
    X = panel[feat_cols].copy()

    # Measure coverage only over the modelling window (rows with a target). The
    # panel index can start decades before WSTS because of long FRED series, and
    # judging a 2005-onwards source against 1948-onwards rows would wrongly drop
    # every timely indicator.
    window = y.notna()
    coverage = X.loc[window].notna().mean()
    X = X.loc[:, coverage[coverage >= min_coverage].index]

    benchmark_cols = [c for c in ("target_pit", *(f"target_pit__lag{k}" for k in (1, 2, 3)))
                      if c in panel.columns]
    for c in benchmark_cols:
        X[c] = panel[c]

    valid = y.notna() & X[benchmark_cols].notna().all(axis=1) if benchmark_cols else y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X = X.replace([np.inf, -np.inf], np.nan)
    return Supervised(
        X=X,
        y=y,
        horizon=horizon,
        feature_names=[c for c in X.columns if c not in benchmark_cols],
        benchmark_cols=benchmark_cols,
    )
