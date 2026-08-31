"""Equity / index price loader (yfinance).

Migrated and simplified from the previous project's `data_collector.py`. Daily
auto-adjusted closes, resampled to month-end. Prices are "published" the day
they print, so their release lag is zero.
"""

from __future__ import annotations

import pandas as pd


def _download_daily(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:  # single ticker
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index)
    return close.sort_index().dropna(how="all")


def load_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    """Return tidy month-end closes: date, series (`px_<ticker>`), value, published."""
    daily = _download_daily(tickers, start_date)
    monthly = daily.resample("ME").last()
    long = (
        monthly.reset_index()
        .melt(id_vars="Date", var_name="ticker", value_name="value")
        .rename(columns={"Date": "date"})
        .dropna(subset=["value"])
    )
    long["series"] = "px_" + long["ticker"].str.replace("^", "", regex=False)
    long["published"] = long["date"]
    return long[["date", "series", "value", "published"]].sort_values(
        ["series", "date"]
    ).reset_index(drop=True)
