"""Matplotlib figures. One shared style so everything looks like one system."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no Tk, safe under background/batch runs

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_STYLE = {
    "figure.figsize": (11, 6),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


_PHASE_COLOR = {
    "Early Cycle": "#bfe3c0",
    "Mid Cycle": "#8ecf8a",
    "Late Cycle": "#f3d18b",
    "Downturn": "#e7a6a6",
}


def cycle_chart(
    phases: pd.DataFrame,
    turns: pd.DataFrame,
    target: pd.Series,
    out_path: str | Path,
    *,
    pit_factor: pd.Series | None = None,
) -> Path:
    """The semiconductor cycle factor with its phases and turning points, over
    the industry's own YoY billings growth."""
    out_path = Path(out_path)
    f = phases["factor"]
    with plt.rc_context({**_STYLE, "figure.figsize": (12, 6.5)}):
        fig, (ax, ax2) = plt.subplots(
            2, 1, sharex=True, height_ratios=[3, 1.4], gridspec_kw={"hspace": 0.08}
        )

        # phase bands
        block = (phases["phase"] != phases["phase"].shift()).cumsum()
        for _, grp in phases.groupby(block):
            ph = grp["phase"].iloc[0]
            if ph in _PHASE_COLOR:
                ax.axvspan(grp.index[0], grp.index[-1], color=_PHASE_COLOR[ph], alpha=0.55, lw=0)

        ax.axhline(0, color="0.5", lw=0.8)
        ax.plot(f.index, f, color="black", lw=1.8, label="cycle factor (smoothed)")
        if pit_factor is not None:
            ax.plot(pit_factor.index, pit_factor, color="#1f77b4", lw=1.0, alpha=0.7,
                    label="cycle factor (real-time / filtered)")
        for _, t in turns.iterrows():
            y = f.get(t["date"])
            if y is None:
                continue
            ax.scatter([t["date"]], [y], marker="v" if t["kind"] == "peak" else "^",
                       color="black", s=45, zorder=5)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55) for c in _PHASE_COLOR.values()]
        leg1 = ax.legend(handles, list(_PHASE_COLOR), ncol=4, frameon=False,
                         loc="upper left", fontsize=8)
        ax.add_artist(leg1)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.set_ylabel("standardised")
        ax.set_title("Semiconductor cycle factor — dynamic factor model + Bry–Boschan phases")

        ax2.axhline(0, color="0.5", lw=0.8)
        ax2.plot(target.index, target * 100, color="#555", lw=1.4)
        ax2.set_ylabel("WSTS 3MMA\nYoY, %")
        ax2.margins(x=0.01)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def nowcast_oos(results: pd.DataFrame, horizon: int, out_path: str | Path) -> Path:
    """Out-of-sample: realised cycle vs each model's walk-forward forecast."""
    out_path = Path(out_path)
    y = results["y_true"] * 100
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(results.index, y, color="black", lw=2.2, label="Realised (WSTS 3MMA YoY)")
        palette = {"pred_ar_benchmark": "0.55", "pred_elasticnet": "#1f77b4",
                   "pred_lightgbm": "#d62728"}
        for col in [c for c in results.columns if c.startswith("pred_")]:
            ax.plot(results.index, results[col] * 100, lw=1.4, alpha=0.9,
                    color=palette.get(col), label=col.removeprefix("pred_"))
        lo, hi = ax.get_ylim()
        ax.fill_between(results.index, lo, hi, where=(results["y_true"] < 0).to_numpy(),
                        color="0.88", step="mid", zorder=0, label="contraction (realised < 0)")
        ax.set_ylim(lo, hi)
        ax.set_title(f"Semiconductor cycle nowcast — walk-forward OOS (h = {horizon} months)")
        ax.set_ylabel("YoY growth, %")
        ax.margins(x=0.01)
        ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path
