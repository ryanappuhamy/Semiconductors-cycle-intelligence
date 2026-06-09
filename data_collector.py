"""Download quarterly fundamentals and daily prices via yfinance."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

EQUIPMENT_TICKERS = ["ASML", "AMAT", "LRCX", "MU", "TXN", "TSM"]
PRICE_TICKERS = ["NVDA", "MU", "MRVL", "VRT", "SOXX", "QQQ"]
HISTORY_START = "2015-01-01"

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


def _statement_df(ticker: yf.Ticker, statement: str, freq: str) -> pd.DataFrame | None:
    if freq == "quarterly":
        if statement == "income":
            for attr in ("quarterly_income_stmt", "quarterly_financials"):
                df = getattr(ticker, attr, None)
                if df is not None and not df.empty:
                    return df
        else:
            for attr in ("quarterly_balance_sheet", "quarterly_balancesheet"):
                df = getattr(ticker, attr, None)
                if df is not None and not df.empty:
                    return df
    else:
        if statement == "income":
            for attr in ("income_stmt", "financials"):
                df = getattr(ticker, attr, None)
                if df is not None and not df.empty:
                    return df
        else:
            for attr in ("balance_sheet", "balancesheet"):
                df = getattr(ticker, attr, None)
                if df is not None and not df.empty:
                    return df
    return None


def _line_item_series(
    ticker: str,
    statement: str,
    row_candidates: tuple[str, ...],
    freq: str,
) -> pd.Series:
    t = yf.Ticker(ticker)
    df = _statement_df(t, statement, freq)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)

    row = _pick_row(df, row_candidates)
    if row is None:
        warnings.warn(f"Row not found in {freq} {statement} for {ticker}")
        return pd.Series(dtype=float, name=ticker)

    series = row.copy()
    series.index = pd.to_datetime(series.index)
    series = series.astype(float).sort_index()
    if freq == "quarterly":
        series = _to_quarter_end_index(series)
    series.name = ticker
    return series


def _quarterly_grid(start: str = HISTORY_START) -> pd.DatetimeIndex:
    end = pd.Timestamp.today().to_period("Q").end_time.normalize()
    begin = pd.Timestamp(start).to_period("Q").end_time.normalize()
    return pd.date_range(start=begin, end=end, freq="QE-DEC")


def _annual_flow_to_quarterly(annual: pd.Series) -> pd.Series:
    """Spread annual flow items (revenue) evenly across the four fiscal quarters."""
    if annual.empty:
        return annual

    records: dict[pd.Timestamp, float] = {}
    for date, value in annual.sort_index().items():
        if pd.isna(value):
            continue
        fy_end = pd.Timestamp(date)
        per_quarter = value / 4
        for i in range(4):
            q_end = (fy_end.to_period("Q") - i).end_time.normalize()
            records[q_end] = per_quarter

    if not records:
        return pd.Series(dtype=float, name=annual.name)

    out = pd.Series(records, name=annual.name).sort_index()
    return out.groupby(level=0).last()


def _annual_stock_to_quarterly(annual: pd.Series) -> pd.Series:
    """Interpolate annual stock items (inventory) onto a quarterly grid."""
    if annual.empty:
        return annual

    annual = annual.sort_index()
    quarter_ends = _quarterly_grid()
    anchor_index = annual.index.union(quarter_ends).sort_values()
    interpolated = annual.reindex(anchor_index).interpolate(method="time")
    result = interpolated.reindex(quarter_ends)
    result.name = annual.name
    return result.dropna(how="all")


def _back_extrapolate_quarterly(series: pd.Series, target_start: str = HISTORY_START) -> pd.Series:
    """Extend a quarterly series backward to target_start using early-sample growth."""
    series = series.dropna().sort_index()
    target = pd.Timestamp(target_start).to_period("Q").end_time.normalize()
    if series.empty or series.index.min() <= target:
        return series

    grid = _quarterly_grid(target_start)
    grid = grid[grid <= series.index.max()]
    filled = series.reindex(grid)

    valid = filled.dropna()
    if len(valid) < 2:
        return series

    lookback = valid.iloc[: min(8, len(valid))]
    if len(lookback) >= 5:
        growth = (lookback.iloc[4] / lookback.iloc[0]) ** 0.25 - 1
    else:
        growth = (lookback.iloc[-1] / lookback.iloc[0]) ** (1 / (len(lookback) - 1)) - 1
    growth = float(np.clip(growth, -0.12, 0.12))

    anchor_date = valid.index[0]
    anchor_value = valid.iloc[0]

    for q_end in grid:
        if q_end >= anchor_date or pd.notna(filled.loc[q_end]):
            continue
        quarters_back = (anchor_date.to_period("Q") - q_end.to_period("Q")).n
        filled.loc[q_end] = anchor_value / ((1 + growth) ** quarters_back)

    filled.name = series.name
    return filled.sort_index()


def _merge_quarterly_annual(
    quarterly: pd.Series,
    annual: pd.Series,
    statement: str,
) -> pd.Series:
    """Prefer reported quarterly values; fill gaps with annual-derived quarterly estimates."""
    if statement == "income":
        from_annual = _annual_flow_to_quarterly(annual)
    else:
        from_annual = _annual_stock_to_quarterly(annual)

    if quarterly.empty:
        merged = from_annual
    elif from_annual.empty:
        merged = quarterly
    else:
        merged = from_annual.copy()
        merged.update(quarterly.dropna())

    merged = _back_extrapolate_quarterly(merged)
    merged = merged[merged.index >= pd.Timestamp(HISTORY_START)].sort_index()
    merged.name = quarterly.name or annual.name
    return merged


def _fetch_merged_line_item(
    ticker: str,
    statement: str,
    row_candidates: tuple[str, ...],
) -> pd.Series:
    quarterly = _line_item_series(ticker, statement, row_candidates, freq="quarterly")
    annual = _line_item_series(ticker, statement, row_candidates, freq="annual")
    return _merge_quarterly_annual(quarterly, annual, statement)


def fetch_quarterly_revenue(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or EQUIPMENT_TICKERS
    frames = [_fetch_merged_line_item(t, "income", REVENUE_ROWS) for t in tickers]
    df = pd.concat(frames, axis=1).sort_index()
    return df.dropna(how="all")


def fetch_quarterly_inventory(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["MU", "TXN"]
    frames = [_fetch_merged_line_item(t, "balance", INVENTORY_ROWS) for t in tickers]
    df = pd.concat(frames, axis=1).sort_index()
    return df.dropna(how="all")


def fetch_daily_prices(
    tickers: list[str] | None = None,
    start: str = HISTORY_START,
) -> pd.DataFrame:
    tickers = tickers or PRICE_TICKERS
    end = pd.Timestamp.today().normalize()

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


def latest_reported_quarter() -> pd.Timestamp | None:
    """Latest quarter with broad quarterly filing coverage (min of per-ticker report dates)."""
    latest_per_ticker: list[pd.Timestamp] = []
    for ticker in EQUIPMENT_TICKERS:
        reported = _line_item_series(ticker, "income", REVENUE_ROWS, freq="quarterly")
        if not reported.empty:
            latest_per_ticker.append(reported.index.max())
    for ticker in ("MU", "TXN"):
        reported = _line_item_series(ticker, "balance", INVENTORY_ROWS, freq="quarterly")
        if not reported.empty:
            latest_per_ticker.append(reported.index.max())
    if not latest_per_ticker:
        return None
    return pd.Timestamp(min(latest_per_ticker))


def collect_all() -> dict[str, pd.DataFrame | pd.Timestamp | None]:
    """Fetch all datasets needed by the cycle intelligence pipeline."""
    return {
        "revenue": fetch_quarterly_revenue(),
        "inventory": fetch_quarterly_inventory(),
        "prices": fetch_daily_prices(),
        "data_as_of": latest_reported_quarter(),
    }
