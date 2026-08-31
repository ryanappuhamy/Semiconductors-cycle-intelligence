"""Turning-point dating on the cycle factor (Bry–Boschan) and the four-phase map.

Bry & Boschan (1971) is the classical rule-based algorithm for dating peaks and
troughs in a monthly series — the same family of logic behind the NBER business
cycle chronology. Our factor is already smooth (a Kalman-smoothed DFM state), so
the pre-smoothing steps of the original are skipped; the censoring rules are kept:

  * a turn is a local extremum over a +/- `window`-month neighbourhood
  * peaks and troughs must alternate (keep the more extreme of a clash)
  * no turn within `edge` months of either end of the sample
  * each phase (turn to turn) lasts at least `min_phase` months
  * each full cycle (peak to peak / trough to trough) lasts at least `min_cycle`

The four industry phases then follow from the turns and the factor's zero line
(the factor is standardised, so 0 = its long-run average):

  trough -> peak  (expansion) :  Early Cycle  below 0,  Mid Cycle  above 0
  peak -> trough  (contraction):  Late Cycle  above 0,  Downturn   below 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ("Early Cycle", "Mid Cycle", "Late Cycle", "Downturn")


def _candidate_turns(x: np.ndarray, window: int) -> list[tuple[int, str]]:
    turns: list[tuple[int, str]] = []
    n = len(x)
    for i in range(window, n - window):
        seg = x[i - window : i + window + 1]
        if x[i] == seg.max() and (x[i] > x[i - 1] or x[i] > x[i + 1]):
            turns.append((i, "peak"))
        elif x[i] == seg.min() and (x[i] < x[i - 1] or x[i] < x[i + 1]):
            turns.append((i, "trough"))
    return turns


def _enforce_alternation(turns: list[tuple[int, str]], x: np.ndarray) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for idx, kind in sorted(turns):
        if out and out[-1][1] == kind:
            prev_idx = out[-1][0]
            keep_new = (x[idx] > x[prev_idx]) if kind == "peak" else (x[idx] < x[prev_idx])
            if keep_new:
                out[-1] = (idx, kind)
            # else: drop the new one
        else:
            out.append((idx, kind))
    return out


def _enforce_min_spacing(
    turns: list[tuple[int, str]], x: np.ndarray, min_phase: int, min_cycle: int
) -> list[tuple[int, str]]:
    changed = True
    while changed and len(turns) > 2:
        changed = False
        # min phase: consecutive turns too close -> drop the shallower one
        for j in range(len(turns) - 1):
            if turns[j + 1][0] - turns[j][0] < min_phase:
                a, b = turns[j], turns[j + 1]
                drop = a if _depth(a, x) < _depth(b, x) else b
                turns = [t for t in turns if t is not drop]
                turns = _enforce_alternation(turns, x)
                changed = True
                break
        if changed:
            continue
        # min cycle: same-type turns (peak..peak / trough..trough) too close
        for j in range(len(turns) - 2):
            if turns[j][1] == turns[j + 2][1] and turns[j + 2][0] - turns[j][0] < min_cycle:
                mid = turns[j + 1]
                turns = [t for t in turns if t is not mid]
                turns = _enforce_alternation(turns, x)
                changed = True
                break
    return turns


def _depth(turn: tuple[int, str], x: np.ndarray) -> float:
    """How extreme a turn is (used to break ties)."""
    return x[turn[0]] if turn[1] == "peak" else -x[turn[0]]


def bry_boschan(
    series: pd.Series,
    *,
    window: int = 5,
    min_phase: int = 5,
    min_cycle: int = 18,
    edge: int = 6,
) -> pd.DataFrame:
    """Dated peaks and troughs of `series`. Columns: date, kind."""
    s = series.dropna()
    x = s.to_numpy(dtype=float)

    turns = _candidate_turns(x, window)
    turns = _enforce_alternation(turns, x)
    turns = [(i, k) for (i, k) in turns if edge <= i < len(x) - edge]
    turns = _enforce_alternation(turns, x)
    turns = _enforce_min_spacing(turns, x, min_phase, min_cycle)

    return pd.DataFrame(
        {"date": [s.index[i] for i, _ in turns], "kind": [k for _, k in turns]}
    )


def _merge_short_runs(phase: pd.Series, min_run: int) -> pd.Series:
    """Absorb any phase run shorter than `min_run` months into the run before it
    (or after it, if it starts the series) — a minimum-duration filter."""
    phase = phase.copy()
    while True:
        block = (phase != phase.shift()).cumsum()
        sizes = phase.groupby(block).transform("size")
        short = sizes < min_run
        if not short.any():
            return phase
        first_block = block[short].iloc[0]
        mask = block == first_block
        pos = np.flatnonzero(mask.to_numpy())
        if pos[0] == 0:  # short run at the very start -> take the next label
            phase.iloc[pos] = phase.iloc[pos[-1] + 1]
        else:
            phase.iloc[pos] = phase.iloc[pos[0] - 1]


def classify_phases(
    factor: pd.Series,
    turns: pd.DataFrame | None = None,
    *,
    level_smooth: int = 3,
    min_run: int = 3,
) -> pd.DataFrame:
    """Per-month phase label from the turning points and the factor's zero line.

    Returns a frame indexed by month with columns: factor, regime
    (expansion/contraction), phase.
    """
    if turns is None:
        turns = bry_boschan(factor)

    f = factor.dropna()
    regime = pd.Series(index=f.index, dtype=object)

    # walk the alternating turns: the segment leading into a peak, and out of a
    # trough, is an expansion; the segment leading into a trough is a contraction.
    marks = list(turns.itertuples(index=False))
    if marks:
        first = marks[0]
        regime.loc[: first.date] = "expansion" if first.kind == "peak" else "contraction"
        for cur, nxt in zip(marks, marks[1:] + [None], strict=False):
            end = nxt.date if nxt is not None else f.index[-1]
            regime.loc[cur.date : end] = (
                "expansion" if cur.kind == "trough" else "contraction"
            )
    else:
        regime.loc[:] = np.where(f.diff().fillna(0) >= 0, "expansion", "contraction")

    # sub-split on the zero line, lightly smoothed so a one-month wiggle across
    # zero does not flip Early<->Mid or Late<->Downturn
    above = f.rolling(level_smooth, min_periods=1).mean() >= 0
    phase = pd.Series(index=f.index, dtype=object)
    phase[(regime == "expansion") & ~above] = "Early Cycle"
    phase[(regime == "expansion") & above] = "Mid Cycle"
    phase[(regime == "contraction") & above] = "Late Cycle"
    phase[(regime == "contraction") & ~above] = "Downturn"
    phase = _merge_short_runs(phase, min_run)

    return pd.DataFrame({"factor": f, "regime": regime, "phase": phase})


def date_cycle(factor: pd.Series, cfg=None) -> dict:
    """Run the full dating pipeline with parameters from `config.params.cycle.dating`.

    Returns {turns, phases, chronology}.
    """
    d = cfg.params.cycle.dating if cfg is not None else None
    turns = bry_boschan(
        factor,
        window=getattr(d, "window", 5),
        min_phase=getattr(d, "min_phase", 5),
        min_cycle=getattr(d, "min_cycle", 18),
        edge=getattr(d, "edge", 6),
    )
    phases = classify_phases(
        factor,
        turns,
        level_smooth=getattr(d, "level_smooth", 3),
        min_run=getattr(d, "min_run", 3),
    )
    return {"turns": turns, "phases": phases, "chronology": phase_chronology(phases)}


def phase_chronology(phases: pd.DataFrame) -> pd.DataFrame:
    """Compress the per-month labels into contiguous phase spans."""
    p = phases["phase"]
    block = (p != p.shift()).cumsum()
    rows = []
    for _, grp in phases.groupby(block):
        rows.append(
            {
                "phase": grp["phase"].iloc[0],
                "start": grp.index[0],
                "end": grp.index[-1],
                "months": len(grp),
            }
        )
    return pd.DataFrame(rows)
