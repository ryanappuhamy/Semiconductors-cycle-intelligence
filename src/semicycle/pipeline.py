"""End-to-end orchestration. Scripts and the CLI are thin wrappers over these."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from .config import Config, load_config
from .features.build import build_panel
from .io.fred import load_fred
from .io.prices import load_prices
from .io.store import Store
from .io.taiwan import load_taiwan_revenue
from .io.wsts import load_wsts
from .nowcast.dataset import make_supervised
from .nowcast.evaluate import scoreboard, walk_forward
from .nowcast.interpret import feature_importance
from .nowcast.models import make_models
from .report.plots import equity_curve, nowcast_oos, strategy_dashboard

# --- ingest ---------------------------------------------------------------


def _summarise(name: str, df: pd.DataFrame) -> dict:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:12]
    return {
        "table": name,
        "rows": int(len(df)),
        "series": int(df["series"].nunique()) if "series" in df else 0,
        "start": str(df["date"].min().date()) if "date" in df and len(df) else None,
        "end": str(df["date"].max().date()) if "date" in df and len(df) else None,
        "sha": h,
    }


def run_ingest(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    s = cfg.sources
    store = Store(cfg.duckdb_path)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "wsts": lambda: load_wsts(
            s.wsts.url,
            actuals_through=s.wsts.actuals_through,
            mom_zscore_alarm=s.wsts.mom_zscore_alarm,
        ),
        "taiwan_revenue": lambda: load_taiwan_revenue(
            s.taiwan_revenue.api_url,
            s.taiwan_revenue.dataset,
            s.taiwan_revenue.companies,
            s.taiwan_revenue.start_date,
            s.taiwan_revenue.core_companies,
        ),
        "prices": lambda: load_prices(
            list(s.prices.benchmarks) + s.prices.universe, s.prices.start_date
        ),
        "fred": lambda: load_fred(s.fred.csv_url, {k: v for k, v in s.fred.series.items()}),
    }

    manifest = []
    for name, fn in jobs.items():
        print(f"[ingest] {name} ...")
        try:
            df = fn()
        except Exception as exc:  # noqa: BLE001 - one bad source shouldn't sink ingest
            print(f"  !! {name} failed: {exc}")
            continue
        if df.empty:
            print(f"  !! {name} returned no rows — skipped")
            continue
        df.to_parquet(cfg.raw_dir / f"{name}.parquet", index=False)
        store.write(name, df)
        row = _summarise(name, df)
        manifest.append(row)
        print(f"  ok: {row['rows']} rows, {row['series']} series, {row['start']}..{row['end']}")

    man = pd.DataFrame(manifest)
    man.attrs["generated"] = datetime.now(timezone.utc).isoformat()
    (cfg.raw_dir / "MANIFEST.json").write_text(
        json.dumps({"generated": man.attrs["generated"], "tables": manifest}, indent=2)
    )
    return man


# --- features ------------------------------------------------------------


def run_features(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    panel = build_panel(cfg)
    cfg.panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cfg.panel_path)
    return panel


# --- nowcast -----------------------------------------------------------


def run_nowcast(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    if not cfg.panel_path.exists():
        run_features(cfg)
    panel = pd.read_parquet(cfg.panel_path)

    cv = cfg.params.cv
    out: dict = {}
    for h in cfg.params.target.horizons:
        data = make_supervised(panel, horizon=h)
        models = make_models(cfg, data.benchmark_cols)
        results = walk_forward(
            data,
            models,
            min_train_months=cv.min_train_months,
            step_months=cv.step_months,
            purge_months=cv.purge_months,
            embargo_months=cv.embargo_months,
            oos_start=cv.oos_start,
        )
        board = scoreboard(results)
        board.round(4).to_csv(cfg.reports_dir / f"scoreboard_h{h}.csv")
        fig = nowcast_oos(results, h, cfg.reports_dir / f"nowcast_oos_h{h}.png")
        results.to_csv(cfg.reports_dir / f"nowcast_oos_h{h}.csv")
        imp = feature_importance(data, cfg.params.models.lightgbm)
        imp.to_csv(cfg.reports_dir / f"feature_importance_h{h}.csv", index=False)
        out[h] = {"results": results, "scoreboard": board, "figure": fig, "importance": imp,
                  "n_features": len(data.feature_names), "n_oos": len(results)}
    return out


# --- strategy --------------------------------------------------------


def run_strategy(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    if not cfg.panel_path.exists():
        run_features(cfg)

    from .cycle.dating import date_cycle
    from .cycle.dfm import fit_cycle_factor
    from .strategy.backtest import grid_returns, monthly_returns, run_backtest
    from .strategy.signal import build_weights
    from .strategy.stats import (
        deflated_sharpe_ratio,
        perf_stats,
        probability_of_backtest_overfitting,
        regime_attribution,
    )

    s = cfg.params.strategy
    panel = pd.read_parquet(cfg.panel_path)
    store = Store(cfg.duckdb_path)
    prices = (
        store.read("prices").assign(date=lambda d: pd.to_datetime(d["date"]))
        .pivot_table(index="date", columns="series", values="value")
    )
    asset_ret = monthly_returns(prices[s.asset])

    weights = build_weights(panel, s)
    bt = run_backtest(weights["weight"], asset_ret, cost_bps=s.cost_bps, benchmark_ret=asset_ret)
    bt = bt.join(weights["signal"])

    grid = grid_returns(panel, asset_ret, s, s.grid)
    n_trials = grid.shape[1]

    strat = bt["strategy_ret"]
    stats = pd.DataFrame(
        {"strategy": perf_stats(strat), "buy_hold_SOXX": perf_stats(bt["asset_ret"])}
    )
    dsr = deflated_sharpe_ratio(
        strat, n_trials=n_trials,
        trial_sharpes=(grid.mean() / grid.std()).to_numpy(),
    )
    pbo = probability_of_backtest_overfitting(grid, n_partitions=s.cscv_partitions)

    phases = date_cycle(fit_cycle_factor(cfg).factor, cfg)["phases"]
    regime = regime_attribution(strat, phases["phase"])

    bt.to_parquet(cfg.root / "data" / "processed" / "backtest.parquet")
    stats.round(4).to_csv(cfg.reports_dir / "strategy_stats.csv")
    regime.round(4).to_csv(cfg.reports_dir / "regime_attribution.csv")
    robust = pd.concat([dsr, pbo]).round(4)
    robust.to_csv(cfg.reports_dir / "strategy_robustness.csv", header=["value"])

    fig1 = equity_curve(bt, cfg.reports_dir / "backtest_equity.png")

    annual_turnover = bt["turnover"].sum() / (len(bt) / 12)
    span = f"{bt.index[0]:%Y-%m}..{bt.index[-1]:%Y-%m}"
    card = (
        f"cycle-timing strategy   (SOXX, {s.cost_bps:.0f} bps, {span})\n"
        f"  ann. return   {stats.loc['ann_return','strategy']:+.1%}   "
        f"(buy&hold {stats.loc['ann_return','buy_hold_SOXX']:+.1%})\n"
        f"  ann. vol      {stats.loc['ann_vol','strategy']:.1%}   "
        f"(buy&hold {stats.loc['ann_vol','buy_hold_SOXX']:.1%})\n"
        f"  Sharpe        {stats.loc['sharpe','strategy']:.2f}   "
        f"(buy&hold {stats.loc['sharpe','buy_hold_SOXX']:.2f})\n"
        f"  max drawdown  {stats.loc['max_drawdown','strategy']:.1%}   "
        f"(buy&hold {stats.loc['max_drawdown','buy_hold_SOXX']:.1%})\n"
        f"  turnover/yr   {annual_turnover:.1f}x     time invested {(bt['weight']>0).mean():.0%}\n"
        f"\n  a risk overlay, not alpha: it cuts vol and drawdown but not below\n"
        f"  buy & hold's Sharpe.\n"
        f"\n  deflated Sharpe ratio  {dsr['deflated_sharpe_ratio']:.2f}   "
        f"(P[true SR>0] over N={n_trials} configs)\n"
        f"  prob. backtest overfit {pbo['pbo']:.2f}   ({int(pbo['n_splits'])} CSCV splits)\n"
    )
    fig2 = strategy_dashboard(bt, phases, regime, card, cfg.reports_dir / "strategy_dashboard.png")

    return {"bt": bt, "stats": stats, "dsr": dsr, "pbo": pbo, "regime": regime,
            "grid": grid, "card": card, "figures": [fig1, fig2]}


# --- report ----------------------------------------------------------


def run_report(cfg: Config | None = None) -> dict:
    """Regenerate the written sector brief from the current pipeline outputs."""
    cfg = cfg or load_config()
    from .cycle.dating import date_cycle
    from .cycle.dfm import fit_cycle_factor
    from .report.brief import generate_brief

    dated = date_cycle(fit_cycle_factor(cfg).factor, cfg)
    strat = run_strategy(cfg)

    def _read(name):
        p = cfg.reports_dir / name
        return pd.read_csv(p, index_col=0) if p.exists() else pd.DataFrame()

    board_h3 = _read("scoreboard_h3.csv")
    weight_now = float(strat["bt"]["weight"].iloc[-1])
    brief = generate_brief(
        dated["phases"], dated["chronology"], board_h3,
        strat["stats"], strat["dsr"], strat["pbo"], weight_now,
    )
    out = cfg.reports_dir / "semiconductor_cycle_brief.txt"
    out.write_text(brief, encoding="utf-8")
    return {"brief": brief, "path": out, "strategy": strat, "phases": dated["phases"]}
