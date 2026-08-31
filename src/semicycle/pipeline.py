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
from .report.plots import nowcast_oos

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
        fig = nowcast_oos(results, h, cfg.reports_dir / f"nowcast_oos_h{h}.png")
        results.to_csv(cfg.reports_dir / f"nowcast_oos_h{h}.csv")
        imp = feature_importance(data, cfg.params.models.lightgbm)
        imp.to_csv(cfg.reports_dir / f"feature_importance_h{h}.csv", index=False)
        out[h] = {"results": results, "scoreboard": board, "figure": fig, "importance": imp,
                  "n_features": len(data.feature_names), "n_oos": len(results)}
    return out
