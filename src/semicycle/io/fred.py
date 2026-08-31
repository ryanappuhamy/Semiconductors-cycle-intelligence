"""FRED loader via the public CSV endpoint (no API key).

Macro breadth for the cycle factor and the nowcast feature set. Some sandboxed
networks block `fred.stlouisfed.org`; callers should treat this source as
best-effort. It works normally from a personal machine or CI.
"""

from __future__ import annotations

import io

import pandas as pd

from ._http import fetch_bytes


def load_fred_series(csv_url: str, code: str, label: str, release_lag_days: int) -> pd.DataFrame:
    """One FRED series as tidy monthly rows: date, series, value, published."""
    raw = fetch_bytes(csv_url.format(code=code)).decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw))
    date_col, value_col = df.columns[0], df.columns[1]
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["series"] = f"fred_{label}"
    df["published"] = df["date"] + pd.Timedelta(days=release_lag_days)
    return df[["date", "series", "value", "published"]].reset_index(drop=True)


def load_fred(csv_url: str, series: dict) -> pd.DataFrame:
    """All configured FRED series, skipping any that fail to download."""
    frames, failed = [], []
    for code, spec in series.items():
        label = spec["label"] if isinstance(spec, dict) else spec.label
        lag = spec["release_lag_days"] if isinstance(spec, dict) else spec.release_lag_days
        try:
            frames.append(load_fred_series(csv_url, code, label, lag))
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code} ({exc})")
    if failed:
        print(f"  [fred] skipped {len(failed)}: {', '.join(failed)}")
    if not frames:
        return pd.DataFrame(columns=["date", "series", "value", "published"])
    return pd.concat(frames, ignore_index=True)
