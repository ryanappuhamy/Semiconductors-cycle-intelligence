"""WSTS Blue Book "Historical Billings Report" loader.

Worldwide semiconductor sales, monthly, by region, back to 1986. The workbook
is a wide year x month grid with a repeating block per year:

    1986
    Americas   <jan> <feb> ... <dec>  <total> <q1..q4>
    Europe     ...
    Japan      ...
    Asia Pacific ...
    Worldwide  ...
    1987
    ...

We reshape it to long form: (date, region, value_musd).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ._http import fetch_bytes

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def _region_key(raw: str) -> str | None:
    s = str(raw).strip().lower()
    if not s or s[0].isdigit():
        return None
    if "world" in s or s in {"total", "ww"}:
        return "worldwide"
    if "americas" in s or s == "us":
        return "americas"
    if "europe" in s:
        return "europe"
    if "japan" in s:
        return "japan"
    if "asia" in s or "pacific" in s:
        return "asia_pacific"
    return None


def _parse_sheet(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    current_year: int | None = None
    for _, row in df.iterrows():
        head = row.iloc[0]
        # a year marker?
        try:
            year = int(float(head))
            if 1980 <= year <= 2100:
                current_year = year
                continue
        except (TypeError, ValueError):
            pass
        if current_year is None:
            continue
        region = _region_key(head)
        if region is None:
            continue
        values = pd.to_numeric(row.iloc[1:13], errors="coerce").to_numpy()
        for m, val in enumerate(values, start=1):
            if pd.isna(val):
                continue
            records.append(
                {
                    "date": pd.Timestamp(current_year, m, 1) + pd.offsets.MonthEnd(0),
                    "region": region,
                    "value_musd": float(val) / 1_000.0,  # workbook is in 1000 US$
                }
            )
    out = pd.DataFrame.from_records(records)
    return out.sort_values(["region", "date"]).reset_index(drop=True)


def _plausibility_warn(actual: pd.DataFrame, z_alarm: float) -> None:
    """Print a warning for months whose worldwide MoM growth is a large outlier
    versus 40 years of history — a guard against corrupt cells in a new release."""
    ww = (
        actual.loc[actual["region"] == "worldwide"]
        .set_index("date")["value_musd"]
        .sort_index()
    )
    g = np.log(ww).diff().dropna()
    # robust scale from the bulk of history, so a corrupt tail can't hide itself
    # by inflating its own standard deviation
    ref = g.iloc[: int(len(g) * 0.9)]
    scale = 1.4826 * (ref - ref.median()).abs().median()
    z = (g - ref.median()) / scale
    bad = z[z.abs() > z_alarm]
    if not bad.empty:
        months = ", ".join(f"{d.date()} (z={z[d]:+.1f})" for d in bad.index)
        print(f"  [wsts] WARNING: implausible MoM moves in worldwide billings: {months}")


def load_wsts(
    url: str,
    *,
    actuals_through: str | None = None,
    mom_zscore_alarm: float = 4.0,
) -> pd.DataFrame:
    """Return tidy monthly WSTS billings.

    Columns: date, series (`wsts_<region>`), value (USD millions), published.
    `published` uses a fixed 35-day release lag (WSTS has no vintage feed).

    `actuals_through` truncates the series at a trusted month (see the note in
    config/sources.yaml about the Jun-2026 book's anomalous 2026 rows).
    """
    blob = fetch_bytes(url)
    xls = pd.ExcelFile(io.BytesIO(blob))
    monthly = _parse_sheet(xls.parse("Monthly Data", header=None))
    monthly["metric"] = "actual"

    _plausibility_warn(monthly, mom_zscore_alarm)

    if actuals_through is not None:
        cutoff = pd.Timestamp(actuals_through)
        n_before = len(monthly)
        monthly = monthly[monthly["date"] <= cutoff]
        dropped = n_before - len(monthly)
        if dropped:
            print(f"  [wsts] capped actuals at {cutoff.date()} (dropped {dropped} rows)")

    if "3MMA" in xls.sheet_names:
        mma = _parse_sheet(xls.parse("3MMA", header=None))
        mma["metric"] = "mma3"
        if actuals_through is not None:
            mma = mma[mma["date"] <= pd.Timestamp(actuals_through)]
        monthly = pd.concat([monthly, mma], ignore_index=True)

    tidy = monthly.rename(columns={"value_musd": "value"})
    tidy["series"] = "wsts_" + tidy["region"] + tidy["metric"].map(
        {"actual": "", "mma3": "_3mma"}
    )
    tidy["published"] = tidy["date"] + pd.Timedelta(days=35)
    return tidy[["date", "series", "value", "published"]].sort_values(
        ["series", "date"]
    ).reset_index(drop=True)
