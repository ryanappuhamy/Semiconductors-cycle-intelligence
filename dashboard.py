"""Multi-panel semiconductor cycle intelligence dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from cycle_classifier import PHASES
from forward_returns import FORWARD_TICKERS, HORIZONS_MONTHS

PHASE_COLORS = {
    "Early Cycle": "#2ecc71",
    "Mid Cycle": "#3498db",
    "Late Cycle": "#f39c12",
    "Downturn": "#e74c3c",
}

INDICATOR_Z_COLS = (
    "equipment_revenue_growth_z",
    "inventory_to_revenue_z",
    "soxx_rel_momentum_z",
)

INDICATOR_LABELS = {
    "equipment_revenue_growth_z": "Equipment Revenue Growth",
    "inventory_to_revenue_z": "Inventory / Revenue (inv.)",
    "soxx_rel_momentum_z": "SOXX vs QQQ Momentum",
}

INDICATOR_LINE_COLORS = {
    "equipment_revenue_growth_z": "#2980b9",
    "inventory_to_revenue_z": "#8e44ad",
    "soxx_rel_momentum_z": "#d35400",
}

PRICE_TICKERS = ["NVDA", "MU", "MRVL"]
PRICE_LINE_COLORS = {"NVDA": "#76b900", "MU": "#1a5276", "MRVL": "#c0392b"}


def _phase_spans(classified: pd.DataFrame, extend_last: bool = True) -> list[tuple]:
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

    end = pd.Timestamp.today().normalize() if extend_last else classified.index[-1]
    spans.append((start, end, current))
    return spans


def _shade_phases(ax: plt.Axes, spans: list[tuple], alpha: float = 0.14) -> None:
    for start, end, phase in spans:
        ax.axvspan(
            start,
            end,
            color=PHASE_COLORS.get(phase, "#bdc3c7"),
            alpha=alpha,
            linewidth=0,
        )


def _status_row(classified: pd.DataFrame, data_as_of: pd.Timestamp | None) -> pd.Series:
    """Latest classification row on or before the last quarter with reported fundamentals."""
    if classified.empty:
        return classified.iloc[-1]
    if data_as_of is None:
        return classified.iloc[-1]
    eligible = classified[classified.index <= data_as_of]
    return eligible.iloc[-1] if not eligible.empty else classified.iloc[-1]


def _build_heatmap_matrix(
    forward_summary: pd.DataFrame,
    classified: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full phase × (ticker, horizon) grid with return % and observation counts."""
    phases_in_summary = set(forward_summary["phase"].unique()) if not forward_summary.empty else set()
    phases_in_classified = (
        set(classified["phase"].unique()) if classified is not None and not classified.empty else set()
    )
    phases_in_data = [p for p in PHASES if p in phases_in_summary or p in phases_in_classified]
    if not phases_in_data:
        phases_in_data = sorted(phases_in_summary | phases_in_classified)

    col_tuples = [(ticker, horizon) for ticker in FORWARD_TICKERS for horizon in HORIZONS_MONTHS]
    col_labels = [f"{t}\n{h}M" for t, h in col_tuples]

    returns = pd.DataFrame(index=phases_in_data, columns=col_labels, dtype=float)
    counts = pd.DataFrame(index=phases_in_data, columns=col_labels, dtype=float)

    lookup = forward_summary.set_index(["phase", "ticker", "horizon_months"])
    for phase in phases_in_data:
        for ticker, horizon in col_tuples:
            label = f"{ticker}\n{horizon}M"
            key = (phase, ticker, horizon)
            if key in lookup.index:
                row = lookup.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                returns.loc[phase, label] = row["avg_forward_return"] * 100
                counts.loc[phase, label] = row["n_obs"]

    counts = counts.fillna(0)
    return returns, counts


def create_dashboard(
    classified: pd.DataFrame,
    forward_summary: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    data_as_of: pd.Timestamp | None = None,
    output_path: str | Path = "semiconductor_cycle_dashboard.png",
) -> Path:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.15)
    fig = plt.figure(figsize=(26, 22))
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[2.0, 1.8, 2.2, 1.3, 2.0],
        hspace=0.42,
        wspace=0.22,
    )

    ax_index = fig.add_subplot(gs[0, :])
    ax_indicator_ts = fig.add_subplot(gs[1, :])
    ax_prices = fig.add_subplot(gs[2, :])
    ax_current_bars = fig.add_subplot(gs[3, 0])
    ax_current_card = fig.add_subplot(gs[3, 1])
    ax_heatmap = fig.add_subplot(gs[4, :])

    phase_spans = _phase_spans(classified)

    # --- Panel 1: Composite cycle index ---
    if not classified.empty:
        _shade_phases(ax_index, phase_spans, alpha=0.16)
        ax_index.plot(
            classified.index,
            classified["cycle_index"],
            color="#2c3e50",
            linewidth=2.8,
            zorder=3,
        )
        ax_index.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        current_phase = classified["phase"].iloc[-1]
        current_idx = classified["cycle_index"].iloc[-1]
        ax_index.scatter(
            [classified.index[-1]],
            [current_idx],
            s=160,
            color=PHASE_COLORS.get(current_phase, "#7f8c8d"),
            zorder=5,
            edgecolors="black",
            linewidths=1.2,
        )
        ax_index.annotate(
            f"Current: {current_phase}",
            xy=(classified.index[-1], current_idx),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    ax_index.set_title("Semiconductor Cycle Index & Historical Phases", fontsize=16, fontweight="bold", pad=12)
    ax_index.set_ylabel("Composite Z-Score Index", fontsize=12)
    ax_index.xaxis.set_major_locator(mdates.YearLocator())
    ax_index.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    index_line = mlines.Line2D([], [], color="#2c3e50", linewidth=2.8, label="Cycle Index")
    phase_patches = [mpatches.Patch(color=c, alpha=0.45, label=p) for p, c in PHASE_COLORS.items()]
    ax_index.legend(handles=[index_line, *phase_patches], loc="upper left", fontsize=10, ncol=3)

    # --- Panel 2: Individual indicator z-scores over time ---
    if not classified.empty:
        _shade_phases(ax_indicator_ts, phase_spans, alpha=0.10)
        for col in INDICATOR_Z_COLS:
            if col not in classified.columns:
                continue
            series = classified[col]
            if col == "inventory_to_revenue_z":
                series = series.clip(-3, 3)
            ax_indicator_ts.plot(
                classified.index,
                series,
                label=INDICATOR_LABELS[col],
                color=INDICATOR_LINE_COLORS[col],
                linewidth=2.2,
                alpha=0.92,
            )
        ax_indicator_ts.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax_indicator_ts.set_ylim(-2.8, 2.8)

    ax_indicator_ts.set_title(
        "Individual Cycle Indicators (Z-Scores; inventory clipped to ±3)",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax_indicator_ts.set_ylabel("Z-Score", fontsize=12)
    ax_indicator_ts.legend(loc="upper left", fontsize=10, ncol=3)
    ax_indicator_ts.xaxis.set_major_locator(mdates.YearLocator())
    ax_indicator_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 3: Indexed prices with phase shading ---
    if prices is not None and not classified.empty:
        available = [t for t in PRICE_TICKERS if t in prices.columns]
        if available:
            plot_start = classified.index.min()
            plot_prices = prices.loc[prices.index >= plot_start, available].dropna(how="all")
            if not plot_prices.empty:
                indexed = (plot_prices / plot_prices.iloc[0]) * 100
                indexed = indexed.clip(lower=1e-3)
                _shade_phases(ax_prices, phase_spans, alpha=0.14)
                for ticker in available:
                    ax_prices.plot(
                        indexed.index,
                        indexed[ticker],
                        label=ticker,
                        color=PRICE_LINE_COLORS.get(ticker, None),
                        linewidth=2.2,
                        zorder=3,
                    )
                ax_prices.set_yscale("log")
                ax_prices.set_title(
                    "NVDA, MU, MRVL — Indexed Price Performance (Log Scale, Base = 100)",
                    fontsize=16,
                    fontweight="bold",
                    pad=12,
                )
                ax_prices.set_ylabel("Indexed Price (log)", fontsize=12)
                ax_prices.legend(loc="upper left", fontsize=11, ncol=3)
                ax_prices.xaxis.set_major_locator(mdates.YearLocator())
                ax_prices.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax_prices.text(
            0.5,
            0.5,
            "Price data unavailable",
            ha="center",
            va="center",
            transform=ax_prices.transAxes,
            fontsize=14,
        )

    # --- Panel 4a: Current indicator levels ---
    z_cols = [c for c in INDICATOR_Z_COLS if c in classified.columns]
    if z_cols and not classified.empty:
        latest = classified[z_cols].iloc[-1]
        labels = [INDICATOR_LABELS[c] for c in z_cols]
        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in latest.values]
        y_pos = np.arange(len(labels))
        ax_current_bars.barh(y_pos, latest.values, color=colors, alpha=0.88, height=0.6)
        ax_current_bars.set_yticks(y_pos)
        ax_current_bars.set_yticklabels(labels, fontsize=11)
        ax_current_bars.axvline(0, color="gray", linestyle="--")
        ax_current_bars.set_xlabel("Z-Score", fontsize=11)
        ax_current_bars.set_title("Current Indicator Levels", fontsize=14, fontweight="bold", pad=10)
        for i, val in enumerate(latest.values):
            ax_current_bars.text(
                val + (0.06 if val >= 0 else -0.06),
                i,
                f"{val:.2f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=11,
                fontweight="bold",
            )

    # --- Panel 4b: Phase classification card ---
    ax_current_card.axis("off")
    if not classified.empty:
        status_row = _status_row(classified, data_as_of)
        as_of_quarter = pd.Timestamp(status_row.name).strftime("%Y-%m-%d")
        card_lines = [
            "CURRENT CYCLE STATUS",
            "",
            f"Phase:       {status_row['phase']}",
            f"Cycle Index: {status_row['cycle_index']:.2f}",
            f"Direction:   {status_row.get('cycle_direction', float('nan')):+.2f} (QoQ)",
            "",
            f"Quarters:    {len(classified)}",
            f"As of:       {as_of_quarter}",
        ]
        ax_current_card.text(
            0.06,
            0.94,
            "\n".join(card_lines),
            transform=ax_current_card.transAxes,
            va="top",
            fontsize=14,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.7",
                facecolor=PHASE_COLORS.get(status_row["phase"], "#ecf0f1"),
                alpha=0.35,
                edgecolor="#2c3e50",
                linewidth=1.5,
            ),
        )
    ax_current_card.set_title("Phase Classification", fontsize=14, fontweight="bold", pad=10)

    # --- Panel 5: Forward return heatmap (all phases × tickers × horizons) ---
    if forward_summary is not None and not forward_summary.empty:
        heat_returns, heat_counts = _build_heatmap_matrix(forward_summary, classified)
        cell_mask = heat_counts == 0
        annot = pd.DataFrame(index=heat_returns.index, columns=heat_returns.columns, dtype=object)
        for row in heat_returns.index:
            for col in heat_returns.columns:
                val = heat_returns.loc[row, col]
                n = heat_counts.loc[row, col]
                if n > 0 and pd.notna(val):
                    annot.loc[row, col] = f"{val:.1f}\n(n={int(n)})"
                else:
                    annot.loc[row, col] = "—"

        sns.heatmap(
            heat_returns,
            annot=annot,
            fmt="",
            cmap="RdYlGn",
            center=0,
            ax=ax_heatmap,
            cbar_kws={"label": "Avg Forward Return (%)", "shrink": 0.85},
            linewidths=0.8,
            linecolor="white",
            mask=cell_mask,
            annot_kws={"fontsize": 9},
        )
        ax_heatmap.set_title(
            "Average Forward Returns by Cycle Phase — 3 / 6 / 12 Month Horizons",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax_heatmap.set_xlabel("Ticker & Horizon", fontsize=12)
        ax_heatmap.set_ylabel("Cycle Phase", fontsize=12)
        ax_heatmap.tick_params(axis="x", labelsize=10)
        ax_heatmap.tick_params(axis="y", labelsize=11, rotation=0)
    else:
        ax_heatmap.text(
            0.5,
            0.5,
            "Insufficient data for forward return heatmap",
            ha="center",
            va="center",
            transform=ax_heatmap.transAxes,
            fontsize=14,
        )

    fig.suptitle(
        "Semiconductor Cycle Intelligence Dashboard",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
