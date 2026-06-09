"""Download quarterly fundamentals and daily prices via yfinance."""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

EQUIPMENT_TICKERS = ["ASML", "AMAT", "LRCX", "MU", "TXN", "TSM"]
PRICE_TICKERS = ["NVDA", "MU", "MRVL", "VRT", "SOXX", "QQQ"]
PRICE_YEARS = 10

REVENUE_ROWS = ("Total Revenue", "Revenue", "Operating Revenue")
INVENTORY_ROWS = ("Inventory", "Inventories")


def _pick_row(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    for name in candidates:
        matches = [idx for idx in df.index if name.lower() in str(idx).lower()]
        if matches:
            return df.loc[matches[0]]
    return None


def _to_quarter_end_index(series: pd.Series) -> pd.Series:
    """Map fiscal report dates onto a common calendar-quarter index."""
    periods = series.index.to_series().dt.to_period("Q")
    grouped = series.groupby(periods).last()
    quarter_ends = grouped.index.to_timestamp(how="end").normalize()
    out = pd.Series(grouped.values, index=quarter_ends, name=series.name)
    return out.sort_index()


def _quarterly_series(ticker: str, statement: str, row_candidates: tuple[str, ...]) -> pd.Series:
    """Extract a quarterly financial line item as a date-indexed series."""
    t = yf.Ticker(ticker)
    if statement == "income":
        df = t.quarterly_income_stmt
        if df is None or df.empty:
            df = t.quarterly_financials
    else:
        df = t.quarterly_balance_sheet
        if df is None or df.empty:
            df = t.quarterly_balancesheet

    if df is None or df.empty:
        warnings.warn(f"No quarterly {statement} data for {ticker}")
        return pd.Series(dtype=float, name=ticker)

    row = _pick_row(df, row_candidates)
    if row is None:
        warnings.warn(f"Row not found in {statement} for {ticker}")
        return pd.Series(dtype=float, name=ticker)

    series = row.copy()
    series.index = pd.to_datetime(series.index)
    series = _to_quarter_end_index(series.astype(float))
    series.name = ticker
    return series


def fetch_quarterly_revenue(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or EQUIPMENT_TICKERS
    frames = [_quarterly_series(t, "income", REVENUE_ROWS) for t in tickers]
    df = pd.concat(frames, axis=1).sort_index()
    return df.dropna(how="all")


def fetch_quarterly_inventory(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["MU", "TXN"]
    frames = [_quarterly_series(t, "balance", INVENTORY_ROWS) for t in tickers]
    df = pd.concat(frames, axis=1).sort_index()
    return df.dropna(how="all")


def fetch_daily_prices(
    tickers: list[str] | None = None,
    years: int = PRICE_YEARS,
) -> pd.DataFrame:
    tickers = tickers or PRICE_TICKERS
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    return prices.dropna(how="all").sort_index()


def collect_all() -> dict[str, pd.DataFrame]:
    """Fetch all datasets needed by the cycle intelligence pipeline."""
    return {
        "revenue": fetch_quarterly_revenue(),
        "inventory": fetch_quarterly_inventory(),
        "prices": fetch_daily_prices(),
    }
