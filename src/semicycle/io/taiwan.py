"""Taiwan monthly revenue loader (FinMind API v4).

Taiwan-listed firms must file revenue within 10 days of month-end, so this is
the fastest hard fundamental for the semiconductor value chain — a month ahead
of WSTS and a full quarter ahead of Western earnings.
"""

from __future__ import annotations

import json

import pandas as pd

from ._http import fetch_bytes


def _fetch_company(api_url: str, dataset: str, data_id: str, start_date: str) -> pd.DataFrame:
    url = f"{api_url}?dataset={dataset}&data_id={data_id}&start_date={start_date}"
    payload = json.loads(fetch_bytes(url))
    if payload.get("status") != 200 or not payload.get("data"):
        return pd.DataFrame(columns=["date", "data_id", "revenue", "published"])
    df = pd.DataFrame(payload["data"])
    # `date` is the filing-period label (first of the revenue month, offset by
    # FinMind's own convention); rebuild the reference month from year/month.
    df["date"] = (
        pd.to_datetime(
            dict(year=df["revenue_year"], month=df["revenue_month"], day=1)
        )
        + pd.offsets.MonthEnd(0)
    )
    pub = pd.to_datetime(df.get("create_time"), errors="coerce")
    df["published"] = pub.fillna(df["date"] + pd.Timedelta(days=11))
    df["data_id"] = data_id
    return df[["date", "data_id", "revenue", "published"]]


def load_taiwan_revenue(
    api_url: str,
    dataset: str,
    companies: dict[str, str],
    start_date: str,
    core_companies: list[str] | None = None,
) -> pd.DataFrame:
    """Return tidy monthly revenue per company plus a value-chain aggregate.

    Columns: date, series (`twrev_<label>` and `twrev_aggregate`), value
    (revenue, TWD), published. The aggregate sums a fixed core set (full history)
    so its growth rate has no composition breaks.
    """
    core = set(core_companies or companies)
    frames, core_frames = [], []
    for data_id, label in companies.items():
        raw = _fetch_company(api_url, dataset, data_id, start_date)
        if raw.empty:
            continue
        raw = raw.assign(series=f"twrev_{label}", value=raw["revenue"].astype(float))
        frames.append(raw[["date", "series", "value", "published"]])
        if data_id in core:
            core_frames.append(raw[["date", "value"]].rename(columns={"value": data_id}))

    if not frames:
        raise ConnectionError("FinMind returned no Taiwan revenue data")

    tidy = pd.concat(frames, ignore_index=True)

    wide = (
        pd.concat([f.set_index("date") for f in core_frames], axis=1)
        .sort_index()
        .dropna()
    )
    agg = pd.DataFrame(
        {
            "date": wide.index,
            "series": "twrev_aggregate",
            "value": wide.sum(axis=1).to_numpy(),
            "published": wide.index + pd.Timedelta(days=11),
        }
    )
    return pd.concat([tidy, agg], ignore_index=True).sort_values(
        ["series", "date"]
    ).reset_index(drop=True)
