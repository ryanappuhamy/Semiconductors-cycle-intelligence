"""Assemble the modelling panel from the raw tidy tables in DuckDB.

Output: one row per month-end, columns =
  * point-in-time features (everything knowable at that month-end), and
  * ``target`` — YoY growth of the 3-month moving average of worldwide WSTS
    billings, dated to its *reference* month and using final (revised) values.

The split matters: features must be causal / as-of, but the target is the thing
we are trying to predict, so it is the best available estimate of what really
happened. `dataset.py` then shifts the target by the forecast horizon and purges
overlapping training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..io.store import Store
from .transforms import as_of_panel, diffusion_index, mma, mom, yoy, zscore_expanding

RAW_TABLES = ["wsts", "taiwan_revenue", "fred", "prices"]


def _load_tidy(store: Store) -> pd.DataFrame:
    frames = []
    for t in RAW_TABLES:
        if t in store.tables():
            df = store.read(t)
            df["date"] = pd.to_datetime(df["date"])
            df["published"] = pd.to_datetime(df["published"])
            frames.append(df[["date", "series", "value", "published"]])
    if not frames:
        raise RuntimeError("no raw tables in store — run scripts/00_ingest.py first")
    return pd.concat(frames, ignore_index=True)


def _monthly_index(tidy: pd.DataFrame) -> pd.DatetimeIndex:
    start = tidy["date"].min().to_period("M").to_timestamp("M")
    end = tidy["date"].max().to_period("M").to_timestamp("M")
    return pd.date_range(start, end, freq="ME")


def build_target(tidy: pd.DataFrame, cfg: Config) -> pd.Series:
    """YoY of the 3MMA of worldwide billings, on the reference-month index."""
    ww = (
        tidy.loc[tidy["series"] == "wsts_worldwide", ["date", "value"]]
        .set_index("date")["value"]
        .sort_index()
    )
    if ww.empty:
        raise RuntimeError("WSTS worldwide series missing from raw data")
    ww = ww.resample("ME").last()
    n = cfg.params.target.smoothing_months
    return yoy(mma(ww, n)).rename("target")


def build_features(tidy: pd.DataFrame, index: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    pit = as_of_panel(tidy, index)
    zmin = cfg.params.features.zscore_expanding_min_periods
    feats: dict[str, pd.Series] = {}

    # --- activity / fundamentals: level series -> growth rates ---------------
    level_prefixes = ("wsts_", "twrev_", "fred_")
    for col in pit.columns:
        if not col.startswith(level_prefixes):
            continue
        if col.endswith("_3mma"):
            continue
        s = pit[col].astype(float)
        feats[f"{col}__yoy"] = yoy(s)
        feats[f"{col}__mom3"] = mom(s, 3)
        feats[f"{col}__yoy_z"] = zscore_expanding(yoy(s), zmin)

    # --- prices: returns and momentum --------------------------------------
    for col in [c for c in pit.columns if c.startswith("px_")]:
        s = pit[col].astype(float)
        feats[f"{col}__ret1"] = s.pct_change(1, fill_method=None)
        for w in cfg.params.features.momentum_windows_months:
            feats[f"{col}__mom{w}"] = s.pct_change(w, fill_method=None)

    # --- legacy relative-momentum signal: SOX vs QQQ, 6m -------------------
    if {"px_SOX", "px_QQQ"}.issubset(pit.columns):
        rel = pit["px_SOX"].pct_change(6, fill_method=None) - pit["px_QQQ"].pct_change(
            6, fill_method=None
        )
        feats["semis_vs_qqq__mom6"] = rel
        feats["semis_vs_qqq__mom6_z"] = zscore_expanding(rel, zmin)

    # --- point-in-time view of the target itself (basis of the AR benchmark) --
    if "wsts_worldwide" in pit.columns:
        n = cfg.params.target.smoothing_months
        tgt_pit = yoy(mma(pit["wsts_worldwide"].astype(float), n))
        feats["target_pit"] = tgt_pit
        for lag in (1, 2, 3, 6, 12):
            feats[f"target_pit__lag{lag}"] = tgt_pit.shift(lag)

    panel = pd.DataFrame(feats, index=index)

    # --- breadth: how many fundamental growth rates are improving ----------
    yoy_cols = [c for c in panel.columns if c.endswith("__yoy")]
    if yoy_cols:
        panel["breadth_diffusion"] = diffusion_index(panel[yoy_cols])

    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def build_panel(cfg: Config) -> pd.DataFrame:
    store = Store(cfg.duckdb_path)
    tidy = _load_tidy(store)
    index = _monthly_index(tidy)
    features = build_features(tidy, index, cfg)
    target = build_target(tidy, cfg).reindex(index)
    panel = features.join(target)
    panel.index.name = "date"
    return panel
