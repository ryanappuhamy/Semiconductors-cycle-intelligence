"""Forward return statistics by semiconductor cycle phase."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_TICKERS = ["NVDA", "MU", "MRVL"]
HORIZONS_MONTHS = (3, 6, 12)
TRADING_DAYS_PER_MONTH = 21


def _forward_return(prices: pd.Series, start: pd.Timestamp, months: int) -> float | None:
    start_px = prices.asof(start)
    if pd.isna(start_px):
        return None

    target = start + pd.DateOffset(months=months)
    if target >= prices.index[-1] - pd.Timedelta(days=5):
        return None

    end_px = prices.asof(target)
    if pd.isna(end_px):
        return None

    return float(end_px / start_px - 1)


def compute_forward_returns(
    classified: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        detail — one row per (quarter, ticker, horizon)
        summary — average forward return by (phase, ticker, horizon)
    """
    tickers = tickers or FORWARD_TICKERS
    rows: list[dict] = []

    for quarter_end, row in classified.iterrows():
        phase = row["phase"]
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            px = prices[ticker].dropna()
            for months in HORIZONS_MONTHS:
                ret = _forward_return(px, quarter_end, months)
                if ret is not None and np.isfinite(ret):
                    rows.append(
                        {
                            "quarter_end": quarter_end,
                            "phase": phase,
                            "ticker": ticker,
                            "horizon_months": months,
                            "forward_return": ret,
                        }
                    )

    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame(
            columns=["phase", "ticker", "horizon_months", "avg_forward_return", "n_obs"]
        )
        return detail, summary

    summary = (
        detail.groupby(["phase", "ticker", "horizon_months"], observed=True)["forward_return"]
        .agg(avg_forward_return="mean", n_obs="count")
        .reset_index()
    )
    return detail, summary
