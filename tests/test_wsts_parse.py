"""WSTS workbook reshaping (year x month grid -> tidy long)."""

import numpy as np
import pandas as pd

from semicycle.io.wsts import _parse_sheet, _region_key


def test_region_key_normalisation():
    assert _region_key("Americas") == "americas"
    assert _region_key("Asia Pacific/All Other") == "asia_pacific"
    assert _region_key("Worldwide") == "worldwide"
    assert _region_key("Total World") == "worldwide"
    assert _region_key("1998") is None
    assert _region_key("Total Year") is None or _region_key("Total Year") == "worldwide"


def test_parse_sheet_reshapes_year_blocks():
    # two year-blocks, one region each, 12 monthly values in cols 1..12
    rows = [
        [np.nan, "January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
        [2000] + [np.nan] * 12,
        ["Worldwide"] + [1000 * (i + 1) for i in range(12)],
        [2001] + [np.nan] * 12,
        ["Worldwide"] + [2000 * (i + 1) for i in range(12)],
    ]
    df = pd.DataFrame(rows)
    tidy = _parse_sheet(df)

    assert set(tidy["region"]) == {"worldwide"}
    assert len(tidy) == 24
    jan_2000 = tidy[tidy["date"] == pd.Timestamp("2000-01-31")]["value_musd"].iloc[0]
    assert jan_2000 == 1.0  # 1000 (1000 US$) -> 1.0 USD millions
    dec_2001 = tidy[tidy["date"] == pd.Timestamp("2001-12-31")]["value_musd"].iloc[0]
    assert dec_2001 == 24.0
    assert tidy["date"].is_monotonic_increasing
