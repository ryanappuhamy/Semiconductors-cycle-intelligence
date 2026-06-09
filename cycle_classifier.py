"""Composite Semiconductor Cycle Index and phase classification."""

from __future__ import annotations

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


def classify_phase(index_level: float, direction: float) -> str:
    """
    Classify a quarter using composite index level and quarter-over-quarter change.

    Early Cycle  — recovering from trough (low level, positive direction)
    Mid Cycle    — expansion (elevated level, non-negative direction)
    Late Cycle   — peak / rollover (elevated level, negative direction)
    Downturn     — contraction (low level, negative direction)
    """
    elevated = index_level >= 0.0
    improving = direction >= 0.0

    if elevated and improving:
        return "Mid Cycle"
    if elevated and not improving:
        return "Late Cycle"
    if not elevated and improving:
        return "Early Cycle"
    return "Downturn"


def classify_series(indicators: pd.DataFrame) -> pd.DataFrame:
    """Add composite index, direction, and phase labels to indicator frame."""
    idx = composite_index(indicators)
    direction = idx.diff()
    phases = [
        classify_phase(level, dir_)
        for level, dir_ in zip(idx, direction, strict=False)
    ]

    out = indicators.copy()
    out["cycle_index"] = idx
    out["cycle_direction"] = direction
    out["phase"] = phases
    # First quarter has no direction — label from level only
    if len(out) > 0:
        first_level = out["cycle_index"].iloc[0]
        out.iloc[0, out.columns.get_loc("phase")] = (
            "Mid Cycle" if first_level >= 0 else "Early Cycle"
        )
    return out.dropna(subset=["cycle_index"])
