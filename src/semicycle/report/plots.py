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


def equity_curve(bt: pd.DataFrame, out_path: str | Path, *, title: str = "") -> Path:
    """Strategy vs buy-and-hold, log scale, with the drawdown underneath."""
    out_path = Path(out_path)
    with plt.rc_context({**_STYLE, "figure.figsize": (11, 6.5)}):
        fig, (ax, ax2) = plt.subplots(
            2, 1, sharex=True, height_ratios=[3, 1], gridspec_kw={"hspace": 0.08}
        )
        ax.plot(bt.index, bt["strategy_cum"], color="#1f77b4", lw=1.8,
                label="cycle-timing strategy")
        base = bt["benchmark_cum"] if "benchmark_cum" in bt else bt["asset_cum"]
        ax.plot(bt.index, base, color="0.45", lw=1.4, label="buy & hold SOXX")
        ax.set_yscale("log")
        ax.set_ylabel("growth of $1 (log)")
        ax.set_title(title or "Cycle-timing strategy vs buy & hold, 10 bps cost")
        ax.legend(frameon=False, loc="upper left")

        dd = bt["strategy_cum"] / bt["strategy_cum"].cummax() - 1
        ax2.fill_between(bt.index, dd * 100, 0, color="#1f77b4", alpha=0.35)
        ax2.set_ylabel("drawdown %")
        ax2.margins(x=0.01)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def strategy_dashboard(
    bt: pd.DataFrame,
    phases: pd.DataFrame,
    regime: pd.DataFrame,
    stats_text: str,
    out_path: str | Path,
) -> Path:
    """Four panels: equity, weight over the cycle phases, per-regime return, stat card."""
    out_path = Path(out_path)
    with plt.rc_context({**_STYLE, "figure.figsize": (13, 8)}):
        fig, axes = plt.subplots(2, 2, gridspec_kw={"hspace": 0.28, "wspace": 0.22})
        a, b, c, d = axes.flat

        a.plot(bt.index, bt["strategy_cum"], color="#1f77b4", lw=1.6, label="strategy")
        base = bt["benchmark_cum"] if "benchmark_cum" in bt else bt["asset_cum"]
        a.plot(bt.index, base, color="0.45", lw=1.3, label="buy & hold")
        a.set_yscale("log")
        a.set_title("Growth of $1 (log)")
        a.legend(frameon=False, fontsize=8)

        ph = phases["phase"].reindex(bt.index)
        block = (ph != ph.shift()).cumsum()
        for _, g in pd.DataFrame({"ph": ph}).groupby(block):
            col = _PHASE_COLOR.get(g["ph"].iloc[0])
            if col:
                b.axvspan(g.index[0], g.index[-1], color=col, alpha=0.5, lw=0)
        b.plot(bt.index, bt["weight"], color="black", lw=1.3)
        b.axhline(1.0, color="0.5", lw=0.8, ls="--")
        b.set_title("Semiconductor weight, over cycle phases")
        b.set_ylabel("weight")

        if not regime.empty:
            bars = c.bar(range(len(regime)), regime["ann_return"] * 100,
                         color=[_PHASE_COLOR.get(p, "0.6") for p in regime.index])
            for rect, n in zip(bars, regime["months"], strict=True):
                c.annotate(f"n={int(n)}", (rect.get_x() + rect.get_width() / 2, 0),
                           ha="center", va="bottom" if rect.get_height() < 0 else "top",
                           fontsize=7, xytext=(0, 3 if rect.get_height() < 0 else -3),
                           textcoords="offset points")
            c.set_xticks(range(len(regime)))
            c.set_xticklabels(regime.index, rotation=20, fontsize=8)
            c.axhline(0, color="0.5", lw=0.8)
            c.set_title("Strategy annualised return by cycle phase")
            c.set_ylabel("%")

        d.axis("off")
        d.text(0, 1, stats_text, va="top", ha="left", family="monospace", fontsize=9)
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
