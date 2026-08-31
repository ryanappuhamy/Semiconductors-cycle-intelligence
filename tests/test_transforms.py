"""The no-look-ahead contract for feature construction."""

import numpy as np
import pandas as pd
import pytest

from semicycle.features.transforms import as_of_panel, mma, yoy, zscore_expanding


@pytest.fixture
def monthly():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    return pd.Series(np.arange(60, dtype=float) + 100, index=idx)


def test_yoy_is_causal(monthly):
    got = yoy(monthly)
    # first 12 values undefined; value at t depends only on t and t-12
    assert got.iloc[:12].isna().all()
    expected = monthly.iloc[12] / monthly.iloc[0] - 1
    assert got.iloc[12] == pytest.approx(expected)


def test_mma_uses_only_trailing(monthly):
    got = mma(monthly, 3)
    assert got.iloc[:2].isna().all()
    assert got.iloc[5] == pytest.approx(monthly.iloc[3:6].mean())


def test_zscore_expanding_no_future(monthly):
    """Truncating the series must not change earlier z-scores."""
    full = zscore_expanding(monthly, min_periods=24)
    truncated = zscore_expanding(monthly.iloc[:40], min_periods=24)
    pd.testing.assert_series_equal(full.iloc[:40], truncated, check_names=False)


def test_as_of_panel_respects_publication_lag():
    # one series, monthly reference dates, each published 40 days later
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    tidy = pd.DataFrame(
        {
            "date": dates,
            "series": "x",
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "published": dates + pd.Timedelta(days=40),
        }
    )
    asof_index = pd.date_range("2020-01-31", periods=6, freq="ME")
    panel = as_of_panel(tidy, asof_index)

    # At 2020-01-31 nothing is published yet (Jan value prints ~Mar 11).
    assert np.isnan(panel.loc["2020-01-31", "x"])
    # By 2020-03-31, only the January figure (published 2020-03-11) is known.
    assert panel.loc["2020-03-31", "x"] == 1.0
    # By 2020-04-30, February (published 2020-04-10) is in.
    assert panel.loc["2020-04-30", "x"] == 2.0


def test_as_of_panel_never_sees_same_month_release():
    dates = pd.date_range("2021-01-31", periods=4, freq="ME")
    tidy = pd.DataFrame(
        {"date": dates, "series": "y", "value": [10.0, 20.0, 30.0, 40.0],
         "published": dates + pd.Timedelta(days=5)},  # published 5 days after month-end
    )
    panel = as_of_panel(tidy, dates)
    # value for month t publishes after month-end t, so column at t must be the
    # PREVIOUS month's figure at most — never t's own.
    assert np.isnan(panel.iloc[0, 0])
    assert panel.iloc[1, 0] == 10.0
