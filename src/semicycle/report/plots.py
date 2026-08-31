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
