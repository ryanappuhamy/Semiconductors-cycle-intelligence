"""Multi-panel semiconductor cycle intelligence dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from cycle_classifier import PHASES

PHASE_COLORS = {
    "Early Cycle": "#2ecc71",
    "Mid Cycle": "#3498db",
    "Late Cycle": "#f39c12",
    "Downturn": "#e74c3c",
}

INDICATOR_LABELS = {
    "equipment_revenue_growth_z": "Equipment Revenue Growth",
    "inventory_to_revenue_z": "Inventory / Revenue (inv.)",
    "soxx_rel_momentum_z": "SOXX vs QQQ Momentum",
}


def _phase_spans(classified: pd.DataFrame) -> list[tuple]:
    if classified.empty:
        return []

    spans = []
    current = classified["phase"].iloc[0]
    start = classified.index[0]

    for dt, phase in classified["phase"].iloc[1:].items():
        if phase != current:
            spans.append((start, dt, current))
            current = phase
            start = dt
    spans.append((start, classified.index[-1], current))
    return spans


def create_dashboard(
    classified: pd.DataFrame,
    forward_summary: pd.DataFrame,
    output_path: str | Path = "semiconductor_cycle_dashboard.png",
) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.2, 1.5], hspace=0.38, wspace=0.28)

    ax_index = fig.add_subplot(gs[0, :])
    ax_indicators = fig.add_subplot(gs[1, 0])
    ax_current = fig.add_subplot(gs[1, 1])
    ax_heatmap = fig.add_subplot(gs[2, :])

    # --- Panel 1: Composite index with phase shading ---
    if not classified.empty:
        ax_index.plot(
            classified.index,
            classified["cycle_index"],
            color="#2c3e50",
            linewidth=2.5,
            label="Semiconductor Cycle Index",
        )
        ax_index.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        for start, end, phase in _phase_spans(classified):
            ax_index.axvspan(
                start,
                end,
                color=PHASE_COLORS.get(phase, "#bdc3c7"),
                alpha=0.18,
            )

        current_phase = classified["phase"].iloc[-1]
        current_idx = classified["cycle_index"].iloc[-1]
        ax_index.scatter(
            [classified.index[-1]],
            [current_idx],
            s=120,
            color=PHASE_COLORS.get(current_phase, "#7f8c8d"),
            zorder=5,
            edgecolors="black",
        )
        ax_index.annotate(
            f"Current: {current_phase}",
            xy=(classified.index[-1], current_idx),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    ax_index.set_title("Semiconductor Cycle Index & Historical Phases", fontweight="bold")
    ax_index.set_ylabel("Composite Z-Score Index")

    index_line = mlines.Line2D([], [], color="#2c3e50", linewidth=2.5, label="Cycle Index")
    phase_patches = [mpatches.Patch(color=c, alpha=0.5, label=p) for p, c in PHASE_COLORS.items()]
    ax_index.legend(handles=[index_line, *phase_patches], loc="upper left", fontsize=9, ncol=2)

    # --- Panel 2: Current indicator levels ---
    z_cols = [c for c in INDICATOR_LABELS if c in classified.columns]
    if z_cols and not classified.empty:
        latest = classified[z_cols].iloc[-1]
        labels = [INDICATOR_LABELS[c] for c in z_cols]
        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in latest.values]
        y_pos = np.arange(len(labels))
        ax_indicators.barh(y_pos, latest.values, color=colors, alpha=0.85)
        ax_indicators.set_yticks(y_pos)
        ax_indicators.set_yticklabels(labels, fontsize=10)
        ax_indicators.axvline(0, color="gray", linestyle="--")
        ax_indicators.set_xlabel("Z-Score")
        ax_indicators.set_title("Current Indicator Levels", fontweight="bold")
        for i, val in enumerate(latest.values):
            ax_indicators.text(
                val + (0.05 if val >= 0 else -0.05),
                i,
                f"{val:.2f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=10,
            )

    # --- Panel 3: Current phase classification card ---
    ax_current.axis("off")
    if not classified.empty:
        latest_row = classified.iloc[-1]
        card_lines = [
            "CURRENT CYCLE STATUS",
            "",
            f"Phase: {latest_row['phase']}",
            f"Cycle Index: {latest_row['cycle_index']:.2f}",
            f"Direction (QoQ): {latest_row.get('cycle_direction', float('nan')):+.2f}",
            "",
            f"As of: {classified.index[-1].strftime('%Y-%m-%d')}",
        ]
        ax_current.text(
            0.05,
            0.95,
            "\n".join(card_lines),
            transform=ax_current.transAxes,
            va="top",
            fontsize=13,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=PHASE_COLORS.get(latest_row["phase"], "#ecf0f1"),
                alpha=0.35,
                edgecolor="#2c3e50",
            ),
        )
    ax_current.set_title("Phase Classification", fontweight="bold")

    # --- Panel 4: Forward return heatmap ---
    if forward_summary is not None and not forward_summary.empty:
        pivot = forward_summary.pivot_table(
            index="phase",
            columns=["ticker", "horizon_months"],
            values="avg_forward_return",
            observed=True,
        )
        pivot = pivot.reindex([p for p in PHASES if p in pivot.index])
        heat_data = pivot * 100

        sns.heatmap(
            heat_data,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            ax=ax_heatmap,
            cbar_kws={"label": "Avg Forward Return (%)"},
            linewidths=0.5,
        )
        ax_heatmap.set_title(
            "Average Forward Returns by Cycle Phase (NVDA, MU, MRVL)",
            fontweight="bold",
        )
        ax_heatmap.set_xlabel("Ticker / Horizon (months)")
        ax_heatmap.set_ylabel("Cycle Phase")
    else:
        ax_heatmap.text(
            0.5,
            0.5,
            "Insufficient data for forward return heatmap",
            ha="center",
            va="center",
            transform=ax_heatmap.transAxes,
        )

    fig.suptitle(
        "Semiconductor Cycle Intelligence Dashboard",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
