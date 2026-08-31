"""Stationary transforms and the point-in-time alignment primitive.

The one rule everything here obeys: a feature value stamped at month ``T`` may
depend only on information whose publication date was ``<= T``. That is enforced
in :func:`as_of_panel`; the plain transforms below are causal (no centered
windows, no forward fill from the future).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def yoy(s: pd.Series) -> pd.Series:
    """Year-over-year growth of a monthly series."""
    return s.pct_change(12, fill_method=None)


def mom(s: pd.Series, n: int = 1) -> pd.Series:
    """n-month growth."""
    return s.pct_change(n, fill_method=None)


def mma(s: pd.Series, n: int = 3) -> pd.Series:
    """Trailing n-month moving average."""
    return s.rolling(n, min_periods=n).mean()


def zscore_expanding(s: pd.Series, min_periods: int = 24) -> pd.Series:
    """Expanding-window z-score: each point standardised against its own past
    only. Ported from the previous project's `cycle_indicators.zscore`, with a
    fixed (not data-dependent) warm-up so it is reproducible."""
    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    return (s - mean) / std.replace(0, np.nan)


def diffusion_index(df: pd.DataFrame) -> pd.Series:
    """Share of columns rising month-on-month (0..1), a breadth gauge."""
    rising = (df.diff() > 0).sum(axis=1)
    valid = df.notna().sum(axis=1)
    return (rising / valid.replace(0, np.nan)).rename("diffusion")


def as_of_panel(tidy: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reshape long (date, series, value, published) data into a wide monthly
    panel that is correct *as of* each month-end in ``index``.

    Panel cell ``[T, s]`` = the most recent observation of series ``s`` whose
    ``published`` date is on or before ``T``. This is what a forecaster sitting
    at month-end ``T`` would actually have on their screen.
    """
    index = pd.DatetimeIndex(index).sort_values()
    asof = pd.DataFrame({"asof": index})
    out = pd.DataFrame(index=index)
    for name, grp in tidy.groupby("series", sort=False):
        g = (
            grp.dropna(subset=["value", "published"])
            .sort_values("published")
            .loc[:, ["published", "value"]]
        )
        if g.empty:
            continue
        merged = pd.merge_asof(asof, g, left_on="asof", right_on="published", direction="backward")
        out[name] = merged["value"].to_numpy()
    return out
