"""Build and normalize semiconductor cycle indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_MAKERS = ["ASML", "AMAT", "LRCX"]
INVENTORY_TICKERS = ["MU", "TXN"]
MOMENTUM_WINDOW = 126  # ~6 months of trading days


def zscore(series: pd.Series, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(3, len(series) // 3)
    rolling_mean = series.expanding(min_periods=min_periods).mean()
    rolling_std = series.expanding(min_periods=min_periods).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def equipment_revenue_growth(revenue: pd.DataFrame) -> pd.Series:
    """Average revenue growth for ASML, AMAT, LRCX (YoY when enough history, else QoQ)."""
    cols = [c for c in EQUIPMENT_MAKERS if c in revenue.columns]
    if not cols:
        return pd.Series(dtype=float, name="equipment_revenue_growth")

    available = revenue[cols].dropna(how="all")
    periods = 4 if len(available) >= 8 else 1

    growth_frames = []
    for col in cols:
        s = revenue[col].sort_index()
        growth = s.pct_change(periods=periods, fill_method=None)
        growth_frames.append(growth)

    avg_growth = pd.concat(growth_frames, axis=1).mean(axis=1, skipna=True)
    avg_growth.name = "equipment_revenue_growth"
    return avg_growth.dropna()


def inventory_to_revenue_ratio(
    revenue: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.Series:
    """Average inventory / revenue for MU and TXN."""
    ratios = []
    for ticker in INVENTORY_TICKERS:
        if ticker not in revenue.columns or ticker not in inventory.columns:
            continue
        rev = revenue[ticker].sort_index()
        inv = inventory[ticker].sort_index()
        aligned = pd.concat([inv, rev], axis=1, keys=["inv", "rev"]).dropna()
        ratio = (aligned["inv"] / aligned["rev"].replace(0, np.nan)).dropna()
        ratio.name = ticker
        ratios.append(ratio)

    if not ratios:
        return pd.Series(dtype=float, name="inventory_to_revenue")

    combined = pd.concat(ratios, axis=1).mean(axis=1, skipna=True)
    combined.name = "inventory_to_revenue"
    return combined.dropna()


def soxx_relative_momentum(prices: pd.DataFrame, window: int = MOMENTUM_WINDOW) -> pd.Series:
    """SOXX total return minus QQQ total return over a rolling window."""
    if "SOXX" not in prices.columns or "QQQ" not in prices.columns:
        return pd.Series(dtype=float, name="soxx_rel_momentum")

    soxx = prices["SOXX"].dropna()
    qqq = prices["QQQ"].dropna()
    aligned = pd.concat([soxx, qqq], axis=1, keys=["soxx", "qqq"]).dropna()

    soxx_ret = aligned["soxx"] / aligned["soxx"].shift(window) - 1
    qqq_ret = aligned["qqq"] / aligned["qqq"].shift(window) - 1
    rel = (soxx_ret - qqq_ret).dropna()
    rel.name = "soxx_rel_momentum"
    return rel


def _align_to_quarter_end(daily_series: pd.Series, quarter_index: pd.DatetimeIndex) -> pd.Series:
    """Sample a daily series at each quarter-end date (last available observation)."""
    if daily_series.empty or len(quarter_index) == 0:
        return pd.Series(dtype=float)

    values = []
    for q in quarter_index:
        window = daily_series.loc[:q]
        values.append(window.iloc[-1] if len(window) else np.nan)

    return pd.Series(values, index=quarter_index, name=daily_series.name)


def build_indicators(
    revenue: pd.DataFrame,
    inventory: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Return quarterly indicators and their z-scores."""
    eq_growth = equipment_revenue_growth(revenue)
    inv_ratio = inventory_to_revenue_ratio(revenue, inventory)
    daily_momentum = soxx_relative_momentum(prices)

    quarter_index = (
        revenue.index.union(inventory.index)
        .union(eq_growth.index)
        .union(inv_ratio.index)
        .sort_values()
        .unique()
    )
    quarter_index = pd.DatetimeIndex(quarter_index)

    momentum_q = _align_to_quarter_end(daily_momentum, quarter_index)

    raw = pd.DataFrame(
        {
            "equipment_revenue_growth": eq_growth.reindex(quarter_index),
            "inventory_to_revenue": inv_ratio.reindex(quarter_index),
            "soxx_rel_momentum": momentum_q,
        }
    )

    # Invert inventory ratio: high inventory/revenue is late-cycle / downturn signal
    raw["inventory_to_revenue_signal"] = -raw["inventory_to_revenue"]

    zscored = pd.DataFrame(
        {
            "equipment_revenue_growth_z": zscore(raw["equipment_revenue_growth"]),
            "inventory_to_revenue_z": zscore(raw["inventory_to_revenue_signal"]),
            "soxx_rel_momentum_z": zscore(raw["soxx_rel_momentum"]),
        },
        index=raw.index,
    )

    combined = pd.concat([raw, zscored], axis=1)
    required = ["equipment_revenue_growth", "inventory_to_revenue", "soxx_rel_momentum"]
    return combined.dropna(subset=required)
