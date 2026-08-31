# Semiconductor Cycle Intelligence

Tracking, nowcasting, and trading the semiconductor demand cycle ("the silicon
cycle") using only free, public data — with the point-in-time discipline and
out-of-sample validation that a research result has to survive to be worth
anything.

This is a ground-up rebuild of an earlier toolkit
([`Semiconductors-cycle-intelligence`](https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence)).
That version had the right economic intuition but stood on ~5 usable quarterly
observations from a single data vendor, a hand-weighted z-score index, and
in-sample averages. This version keeps the intuition and replaces the
foundation. See [Relationship to the previous project](#relationship-to-the-previous-project).

---

## The cycle, and why it is tradable

Semiconductor demand moves in multi-year cycles driven by capacity investment,
channel inventory build/drawdown, and end-market demand. Three signals tend to
lead or confirm the turns:

1. **Supply-side investment** — equipment-maker and foundry revenue growth marks
   capacity expansion (early/mid cycle).
2. **Inventory pressure** — inventory rising relative to sales marks oversupply
   (late cycle / downturn).
3. **Market pricing** — semiconductor equities outperforming the broad market
   reflects elevated expectations.

The industry taxonomy has four phases — **Early / Mid / Late / Downturn** —
defined on the *level* and *direction* of the cycle. This project makes that
chronology data-driven and asks whether it carries information a systematic
equity strategy can use.

---

## Architecture

```
config/            sources.yaml (URLs, tickers, release lags) + params.yaml (model/CV/strategy)
src/semicycle/
  io/              one loader per source -> tidy (date, series, value, published)
  features/        point-in-time alignment (as_of_panel) + stationary transforms -> panel
  cycle/           [Module 2] mixed-frequency dynamic factor model + Bry-Boschan dating
  nowcast/         supervised dataset, model zoo, purged/embargoed walk-forward CV
  strategy/        [Module 3] cycle-driven equity strategy + honest backtest stats
  report/          figures + brief
scripts/           00_ingest -> 01_build_features -> 02_fit_cycle -> 03_nowcast -> 04_backtest -> 05_report
data/              raw/ (immutable pulls) -> processed/ (DuckDB + panel.parquet)
```

DuckDB is the analytical store: one file, no server, SQL over Parquet.

---

## Data sources (all free, no paid terminal)

| Source | Series | History | Role |
|---|---|---|---|
| **WSTS** Blue Book Historical Billings Report | Worldwide semiconductor sales, monthly, by region | 1986– | Nowcast **target**; coincident cycle input |
| **FinMind** — Taiwan monthly revenue | TSMC, UMC, ASE, MediaTek, Nanya, … (mandatory 10-day filings) | 2005– | Fastest hard fundamental; ~1 month ahead of WSTS |
| **FRED** (public CSV, no key) | IP semiconductors, new orders, inventory/sales, semi PPI, unemployment, Nasdaq | 1948– | Macro breadth |
| **yfinance** | `^SOX`, `SOXX`, `SMH`, `QQQ`, `^GSPC` + a 15-name equity universe | 2004– | Market-momentum features; strategy universe |

**Point-in-time.** Every source has a known `release_lag_days` (WSTS ~35, Taiwan
~11, FRED per-series). `features.as_of_panel` builds the feature matrix so that
the row dated month `T` contains only values whose publication date was `≤ T` —
what a forecaster actually had on screen at month-end `T`. The prediction target
is the final (revised) value, dated to its reference month; only the *inputs*
are restricted.

> **Note on the Jun-2026 WSTS book.** Its 2026 rows imply an H1-2026 worldwide
> run-rate roughly 2x the known market size (monthly MoM jumps of +20–26%,
> unprecedented in 40 years). The loader caps WSTS actuals at 2025-12 and prints
> a plausibility warning. Revisit when a newer book is published.

---

## Validation

**Expanding-window walk-forward with purge and embargo.** For each out-of-sample
month `t`, retrain on all data old enough that its target window cannot overlap
`t`'s:

```
train rows d  with  d + horizon + purge + embargo  ≤  t     (months)
```

`purge` drops rows whose label interval `[d, d+h]` reaches into the test month
(López de Prado); `embargo` adds a further gap against slow autocorrelation.

**Benchmark.** An autoregression expressed only in point-in-time terms (the
stale-but-real WSTS target and its lags). Feature-based models have to beat it
out-of-sample, not in-sample.

**Metrics.** MAE, RMSE, skill vs the AR benchmark, sign accuracy (accelerating
vs contracting), turning-point accuracy, correlation.

---

## Results (Module 1)

240 out-of-sample months (2006–2025), expanding walk-forward with purge + embargo.
Full scoreboards in `reports/scoreboard_h*.csv`, figures in
`reports/nowcast_oos_h*.png`, discussion in [NOTES_IT.md](NOTES_IT.md).

**On full-sample error, the point-in-time autoregression wins at every horizon** —
as it should, for a 3-month-average YoY series that is ~0.95 autocorrelated. The
feature models get close but do not beat it on mean error. A model that *crushed*
this benchmark would be a leak, not a discovery.

**The feature models' edge is at the inflections** — the third of months where the
cycle is actually moving (`|Δ|` above its 67th percentile):

| Horizon | Metric | AR benchmark | LightGBM |
|---|---|---|---|
| h = 3 | MAE (turning-point months) | 0.107 | **0.103** (−3.5%) |
| h = 3 | direction-of-change accuracy | 0.42 | **0.47** |
| h = 6 | MAE (turning-point months) | 0.140 | **0.132** (−5.6%) |

On quiet months AR wins by simply persisting the last value, which keeps its
aggregate MAE low — but a cycle-timing strategy only trades around the turns.
Feature importance is dominated by the WSTS regional structure (Asia-Pacific YoY)
and Nasdaq / inventory-to-sales; the Taiwan revenue feed adds *timeliness*, not
orthogonal information (WSTS "Asia Pacific" already carries it).

---

## Reproduce

```bash
pip install -e ".[dev]"      # Python 3.10+
make ingest                  # WSTS + Taiwan + FRED + prices -> data/raw + DuckDB
make features                # -> data/processed/panel.parquet
make nowcast                 # walk-forward scoreboard + reports/nowcast_oos_h*.png
make test                    # anti-leakage + smoke tests
make lint                    # ruff
```

No API keys. FRED is best-effort — some sandboxed networks block
`fred.stlouisfed.org`; the pipeline continues without it and still trains on
WSTS + Taiwan + market data.

---

## Roadmap

- **Module 2 — the cycle factor.** `cycle/dfm.py`: a Stock–Watson mixed-frequency
  dynamic factor model (`statsmodels DynamicFactorMQ`) over billings, Taiwan
  revenue, FRED activity and equity momentum, giving one latent coincident
  cycle index. `cycle/dating.py`: Bry–Boschan turning-point dating → the
  Early/Mid/Late/Downturn chronology. The factor then feeds back as a nowcast
  feature.
- **Module 3 — the strategy.** `strategy/`: cycle level + momentum → a timing
  signal on `^SOX`/`SMH` and a cross-sectional tilt across the universe. Monthly
  rebalance, explicit costs, and statistics that quantify overfitting risk:
  Sharpe, **deflated Sharpe ratio**, **probability of backtest overfitting**
  (CSCV), turnover, per-regime performance. Migrates the old project's dashboard
  and written brief.
- **Module 4.** Real-time data vintages (ALFRED), a static HTML dashboard,
  final write-up.

---

## Relationship to the previous project

| Previous | Here |
|---|---|
| 3-pillar intuition (equipment capex, inventory, relative momentum) | **Kept** as feature families |
| 4-phase taxonomy + level×direction rules | **Kept**; becomes the output of Bry–Boschan dating on the factor |
| yfinance daily prices | **Migrated** to `io/prices.py` (+ month-end resampling) |
| expanding-window `zscore` | **Migrated** to `features/transforms.py` with anti-leakage tests |
| dashboard, AI brief | **Migrated** in Module 3 |
| yfinance *fundamentals* + synthetic annual→quarterly backfill / back-extrapolation | **Removed** — replaced by WSTS (40y) + Taiwan revenue (20y) + FRED |
| hand-weighted z-score composite | **Replaced** by an estimated dynamic factor |
| in-sample forward-return averages (n = 1–3 per phase) | **Replaced** by purged walk-forward nowcast + a cost-aware backtest |
| single scripts | **Replaced** by a package, config, tests, CI |

---

## Disclaimer

Educational and research use only. Not investment advice. Free data is delayed,
revised, and incomplete; results are illustrative.
