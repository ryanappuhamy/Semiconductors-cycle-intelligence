"""Bry–Boschan turning-point dating and the four-phase map."""

import numpy as np
import pandas as pd

from semicycle.cycle.dating import (
    PHASES,
    bry_boschan,
    classify_phases,
    phase_chronology,
)


def _sine_cycle(n_cycles: int = 4, period: int = 48, noise: float = 0.05) -> pd.Series:
    n = n_cycles * period
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * np.arange(n) / period) + rng.normal(0, noise, n)
    return pd.Series(x, index=idx)


def test_bry_boschan_alternates_and_counts():
    s = _sine_cycle(n_cycles=4, period=48)
    turns = bry_boschan(s)
    kinds = turns["kind"].tolist()
    # peaks and troughs strictly alternate
    assert all(a != b for a, b in zip(kinds, kinds[1:], strict=False))
    # ~4 cycles -> roughly 7-9 turns, none at the very edges
    assert 6 <= len(turns) <= 10
    assert turns["date"].min() > s.index[3]
    assert turns["date"].max() < s.index[-3]


def test_bry_boschan_min_cycle_enforced():
    s = _sine_cycle(n_cycles=6, period=40)
    turns = bry_boschan(s, min_cycle=18)
    # distance between successive same-type turns >= min_cycle
    for kind in ("peak", "trough"):
        d = turns.loc[turns["kind"] == kind, "date"].sort_values()
        months = d.diff().dropna().dt.days / 30.44
        assert (months >= 17).all()


def test_classify_phases_labels_and_no_short_runs():
    s = _sine_cycle(n_cycles=4, period=48)
    turns = bry_boschan(s)
    ph = classify_phases(s, turns, min_run=3)
    assert set(ph["phase"].dropna().unique()).issubset(set(PHASES))
    chron = phase_chronology(ph)
    assert (chron["months"] >= 3).all()
    # a clean expansion should pass Early -> Mid before the peak
    assert {"Early Cycle", "Mid Cycle", "Late Cycle", "Downturn"} & set(ph["phase"])


def test_phase_direction_matches_regime():
    s = _sine_cycle()
    ph = classify_phases(s)
    exp = ph["regime"] == "expansion"
    # expansions are Early or Mid, contractions are Late or Downturn
    assert ph.loc[exp, "phase"].isin(["Early Cycle", "Mid Cycle"]).all()
    assert ph.loc[~exp, "phase"].isin(["Late Cycle", "Downturn"]).all()
