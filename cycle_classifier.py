"""Composite Semiconductor Cycle Index and phase classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ("Early Cycle", "Mid Cycle", "Late Cycle", "Downturn")
Z_COLUMNS = (
    "equipment_revenue_growth_z",
    "inventory_to_revenue_z",
    "soxx_rel_momentum_z",
)
WEIGHTS = {
    "equipment_revenue_growth_z": 0.40,
    "inventory_to_revenue_z": 0.30,
    "soxx_rel_momentum_z": 0.30,
}


def composite_index(indicators: pd.DataFrame) -> pd.Series:
    """Weighted average of z-scored indicators."""
    available = [c for c in Z_COLUMNS if c in indicators.columns]
    if not available:
        return pd.Series(dtype=float, name="cycle_index")

    total_weight = sum(WEIGHTS[c] for c in available)
    index = sum(indicators[c] * WEIGHTS[c] for c in available) / total_weight
    index.name = "cycle_index"
    return index


def classify_phase(
    index_level: float,
    direction: float,
    prior_level: float | None = None,
) -> str:
    """
    Classify a quarter using composite index level, QoQ direction, and prior level.

    Late Cycle uses the *prior* quarter's elevation when direction turns negative,
    because sharp rollovers often pull the index below zero in the same quarter.

    Early Cycle  — recovering from trough (prior below 0, direction positive)
    Mid Cycle    — expansion (elevated and improving)
    Late Cycle   — peak / rollover (elevated prior level, negative direction)
    Downturn     — contraction (not elevated, negative direction)
    """
    if prior_level is None or not np.isfinite(prior_level):
        prior_level = index_level

    if not np.isfinite(direction):
        return "Mid Cycle" if index_level >= 0 else "Early Cycle"

    was_elevated = prior_level >= 0.0
    is_elevated = index_level >= 0.0
    improving = direction >= 0.0

    if was_elevated and not improving:
        return "Late Cycle"
    if is_elevated and improving:
        return "Mid Cycle"
    if not was_elevated and improving:
        return "Early Cycle" if index_level < 0 else "Mid Cycle"
    return "Downturn"


def classify_series(indicators: pd.DataFrame) -> pd.DataFrame:
    """Add composite index, direction, and phase labels to indicator frame."""
    idx = composite_index(indicators)
    direction = idx.diff()
    prior = idx.shift(1)

    phases = [
        classify_phase(level, dir_, prior_level=prev)
        for level, dir_, prev in zip(idx, direction, prior, strict=False)
    ]

    out = indicators.copy()
    out["cycle_index"] = idx
    out["cycle_direction"] = direction
    out["phase"] = phases
    return out.dropna(subset=["cycle_index"])
