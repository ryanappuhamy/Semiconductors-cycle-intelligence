"""Generate a structured semiconductor sector brief via Anthropic Claude."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

MODEL = "claude-sonnet-4-20250514"
MODEL_FALLBACKS = (
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
)


def _build_context(
    classified: pd.DataFrame,
    forward_summary: pd.DataFrame,
) -> dict:
    latest = classified.iloc[-1]
    context = {
        "as_of_quarter": str(classified.index[-1].date()),
        "current_phase": latest["phase"],
        "cycle_index": round(float(latest["cycle_index"]), 3),
        "cycle_direction_qoq": round(float(latest.get("cycle_direction", 0)), 3),
        "indicator_z_scores": {
            "equipment_revenue_growth": round(float(latest.get("equipment_revenue_growth_z", 0)), 3),
            "inventory_to_revenue": round(float(latest.get("inventory_to_revenue_z", 0)), 3),
            "soxx_relative_momentum": round(float(latest.get("soxx_rel_momentum_z", 0)), 3),
        },
        "recent_phase_history": [
            {"quarter": str(idx.date()), "phase": row["phase"], "index": round(float(row["cycle_index"]), 3)}
            for idx, row in classified.tail(6).iterrows()
        ],
    }

    if forward_summary is not None and not forward_summary.empty:
        context["historical_forward_returns"] = (
            forward_summary.assign(
                avg_forward_return_pct=lambda d: (d["avg_forward_return"] * 100).round(2)
            )[["phase", "ticker", "horizon_months", "avg_forward_return_pct", "n_obs"]]
            .to_dict(orient="records")
        )

    return context


def generate_brief(
    classified: pd.DataFrame,
    forward_summary: pd.DataFrame,
    api_key: str | None = None,
    output_path: str | Path = "semiconductor_cycle_brief.txt",
) -> str:
    """
    Call Claude to produce a structured sector brief.
    Requires ANTHROPIC_API_KEY in the environment if api_key is not passed.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        brief = _fallback_brief(classified)
        Path(output_path).write_text(brief, encoding="utf-8")
        return brief

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Install anthropic: pip install anthropic") from exc

    context = _build_context(classified, forward_summary)
    prompt = f"""You are a semiconductor sector strategist. Using the cycle data below, write a concise,
structured brief for institutional investors.

DATA:
{json.dumps(context, indent=2)}

Structure your response with these exact section headers:

## Current Phase Assessment
## Key Indicator Readings
## Historical Context
## Key Risks
## 6-12 Month Outlook
## Positioning Implications (NVDA, MU, MRVL)

Be specific, data-driven, and avoid generic filler. Reference the actual z-scores and phase classification."""

    client = anthropic.Anthropic(api_key=api_key)
    message = None
    last_error: Exception | None = None
    for model in MODEL_FALLBACKS:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except anthropic.NotFoundError as exc:
            last_error = exc
            continue

    if message is None:
        raise RuntimeError(
            f"No supported Claude model available. Tried: {', '.join(MODEL_FALLBACKS)}"
        ) from last_error

    brief = message.content[0].text
    Path(output_path).write_text(brief, encoding="utf-8")
    return brief


def _fallback_brief(classified: pd.DataFrame) -> str:
    latest = classified.iloc[-1]
    return f"""## Semiconductor Cycle Brief (Local Fallback)

ANTHROPIC_API_KEY not set — showing template from computed data.

## Current Phase Assessment
Phase: {latest['phase']}
Cycle Index: {latest['cycle_index']:.2f}
Direction (QoQ): {latest.get('cycle_direction', float('nan')):+.2f}

## Key Indicator Readings
- Equipment Revenue Growth Z: {latest.get('equipment_revenue_growth_z', float('nan')):.2f}
- Inventory/Revenue Z (inverted): {latest.get('inventory_to_revenue_z', float('nan')):.2f}
- SOXX vs QQQ Momentum Z: {latest.get('soxx_rel_momentum_z', float('nan')):.2f}

Set ANTHROPIC_API_KEY to enable Claude-generated analysis.
"""
