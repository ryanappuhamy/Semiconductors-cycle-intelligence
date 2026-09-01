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
  cycle/           dynamic factor model (latent cycle index) + Bry-Boschan phase dating
  nowcast/         supervised dataset, model zoo, purged/embargoed walk-forward CV
  strategy/        cycle-timing signal + cost-aware backtest + deflated Sharpe / PBO
  report/          figures + written brief
scripts/           00_ingest -> 01_build_features -> 02_fit_cycle -> 03_nowcast
                   -> 04_backtest -> 05_report -> 06_realtime -> 07_dashboard
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

## The cycle factor

The previous project combined three indicators with fixed hand-picked weights
(0.40 / 0.30 / 0.30). Here the common component is **estimated** — a single
latent factor with AR(2) dynamics drives a set of coincident indicators, each
with its own loading and idiosyncratic noise:

```
x_it = λ_i · f_t + e_it          f_t = a₁ f_{t-1} + a₂ f_{t-2} + u_t
```

Fitted by EM (`statsmodels DynamicFactorMQ`), which also handles the **ragged
edge** — the indicators end in different months because they publish with
different lags. Inputs (`config/params.yaml` → `cycle:`): the four WSTS regional
billings (YoY), the Taiwan value-chain revenue (YoY), and 12-month SOX momentum —
all coincident views of *global* semiconductor demand. US IP and new-orders were
tried and dropped: US fab output is a capacity-constrained slice that stayed flat
through the 2021 boom and pulled the factor off the cycle.

The fitted factor (1987–) correlates **0.95** with WSTS 3MMA-YoY and lines up
with the record — 2000 peak, 2001 bust, GFC, 2019 memory glut, 2021 shortage
boom, 2023 trough. **`cycle.dating`** then applies the classical **Bry–Boschan**
turning-point algorithm (alternating local extrema, minimum 5-month phases and
18-month cycles, edge censoring) and maps peaks/troughs + the factor's zero line
onto the four industry phases:

```
trough → peak  (expansion) :  Early Cycle below 0,  Mid Cycle above 0
peak → trough  (contraction):  Late Cycle  above 0,  Downturn  below 0
```

`make cycle` writes `reports/cycle_factor.png` and `reports/cycle_chronology.csv`
(35 phase spans, ~14-month average — a ~4-year full cycle). Current reading:
**Mid Cycle**, factor **+1.7**.

For the nowcast the factor re-enters as a **pseudo-real-time** feature: built from
point-in-time inputs (`as_of_panel`) and taken as the Kalman-*filtered* state, so
the factor value at month `T` uses only data available by `T` (model parameters
are still full-sample — the standard pseudo-real-time approximation).

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

## The strategy and its honesty checks

`strategy/` turns the point-in-time cycle signal into a monthly target weight on
SOXX (`clip(base + gain·s, 0, max)`), runs it through a backtest with an explicit
timing convention and 10 bps cost, and then subjects it to the two statistics
that matter after a parameter search:

- **Deflated Sharpe ratio** (Bailey & López de Prado 2014) — the probability the
  true Sharpe is > 0 *after* accounting for having picked the best of `N`
  configurations. Uses the return series' skew and kurtosis and the spread of
  Sharpes across the grid.
- **Probability of backtest overfitting** (CSCV; Bailey, Borwein, López de Prado,
  Zhu 2017) — split the sample into `S` blocks; for every way of choosing `S/2`
  as in-sample, take the config with the best in-sample Sharpe and check where it
  ranks out-of-sample. PBO = share of splits where that config lands below the
  out-of-sample median.

The 16-config grid (`strategy.grid` in `config/params.yaml`) is the search these
two account for. `strategy.stats.regime_attribution` then breaks the strategy's
return down by the Module 2 cycle phase.

---

## Results

240 out-of-sample months (2006–2025), expanding walk-forward with purge + embargo.
Full scoreboards in `reports/scoreboard_h*.csv`, figures in
`reports/nowcast_oos_h*.png`, discussion in [NOTES_IT.md](NOTES_IT.md).

**Module 1 (indicators only).** On full-sample error the point-in-time
autoregression won at every horizon — as it should, for a 3-month-average YoY
series that is ~0.95 autocorrelated. The feature models only edged it at the
inflections (LightGBM −3.5% turning-point MAE at h = 3). A model that *crushed*
this benchmark would be a leak, not a discovery.

**Module 2 (add the latent cycle factor).** Feeding the DFM factor and its
6-month change into the feature set moves the feature models past the benchmark:

| Horizon | Model | MAE | Skill vs AR | MAE at turning points | vs AR |
|---|---|---|---|---|---|
| h = 0 | ElasticNet | 0.0314 | **+1.9%** | 0.045 | −5% |
| h = 3 | LightGBM | 0.0722 | **+3.5%** (was −9.3%) | 0.091 | **−15%** |
| h = 6 | LightGBM | 0.0944 | **+6.7%** (was −0.6%) | 0.125 | **−11%** |

`cycle_factor__chg6` is the single most important LightGBM feature at h = 6
(25% of gain) and third at h = 3. The estimated latent factor carries cycle
information that the raw indicators, taken individually, do not — which is the
whole point of estimating it.

**Module 3 (the strategy).** A cycle-timing overlay on SOXX — target weight
`clip(1.0 + 0.4·s, 0, 1.25)` from the point-in-time cycle signal `s`, monthly
rebalance, 10 bps cost. 2005–2025, vs buy-and-hold:

| | strategy | buy & hold SOXX |
|---|---|---|
| annualised return | +10.9% | +15.5% |
| annualised vol | 21.6% | 25.0% |
| **Sharpe** | **0.59** | **0.70** |
| max drawdown | −54% | −60% |

The overlay cuts vol and drawdown but **not** below buy-and-hold's Sharpe — the
same ~0.1 Sharpe gap holds across every asset (SMH, an equal-weight basket) and
every sub-period. It is a **risk overlay, not an alpha source**: semiconductor
equities are forward-looking and largely price the fundamental cycle before it
reaches billings. Strategy return by phase is +23% / +20% annualised in Early /
Late Cycle and −8% in Downturn — the de-risking works directionally.

The **deflated Sharpe ratio** (0.99 = P[true Sharpe > 0] after N = 16 grid
configs) and the **probability of backtest overfitting** (0.32, CSCV) confirm the
positive Sharpe is real but modest, and that chasing the best-in-grid config
would be overfitting. `reports/strategy_dashboard.png`,
`reports/semiconductor_cycle_brief.txt`.

**Module 4 (real-time checks).** Timing the overlay off the *forward nowcast*
(the walk-forward's real-time 3-month-ahead forecast) instead of the coincident
factor lifts the Sharpe from 0.83 to **0.88** and the return by ~3 pp/yr — a
coincident cycle signal really is too late for equity timing. Still short of
buy-and-hold (0.997) on 2009–2025, a window with almost no drawdown to protect
against. Separately, the pseudo-real-time factor (Module 2, parameters fixed)
correlates **0.94** with a fully recursive one (parameters re-estimated every
month) — the shortcut holds. `make dashboard` bundles it all into a
self-contained `reports/dashboard.html`.

---

## Reproduce

```bash
pip install -e ".[dev]"      # Python 3.10+
make ingest                  # WSTS + Taiwan + FRED + prices -> data/raw + DuckDB
make features                # -> data/processed/panel.parquet (incl. the cycle factor)
make cycle                   # DFM factor + Bry-Boschan phases -> reports/cycle_factor.png
make nowcast                 # walk-forward scoreboard + reports/nowcast_oos_h*.png
make backtest                # cycle-timing strategy: stats, deflated Sharpe, PBO
make report                  # written sector brief
make realtime                # nowcast-timing vs factor-timing; recursive-factor check
make dashboard               # -> reports/dashboard.html (self-contained)
make test                    # anti-leakage + backtest-mechanics + smoke tests
make lint                    # ruff
```

`make report` writes a local template brief; set `ANTHROPIC_API_KEY` for a
Claude-written one (the previous project's `ai_brief.py`, migrated).

No API keys. FRED is best-effort — some sandboxed networks block
`fred.stlouisfed.org`; the pipeline continues without it and still trains on
WSTS + Taiwan + market data.

---

## Roadmap

- **Module 2 — the cycle factor. Done** (see [The cycle factor](#the-cycle-factor)).
- **Module 3 — the strategy. Done** (see [Results](#results)). Cost-aware
  cycle-timing backtest with deflated Sharpe / PBO; the honest finding is that
  it is a risk overlay, not alpha.
- **Module 4 — real-time checks. Done** (see [Results](#results)). Nowcast-based
  timing, a fully recursive factor, and a self-contained HTML dashboard.
- **Next.** Real-time data vintages (ALFRED) for a fully point-in-time feature
  set; a cross-sectional stock-level tilt alongside the index-timing overlay.

---

## Relationship to the previous project

| Previous | Here |
|---|---|
| 3-pillar intuition (equipment capex, inventory, relative momentum) | **Kept** as feature families |
| 4-phase taxonomy + level×direction rules | **Kept** — now the output of Bry–Boschan dating on the factor (`cycle/dating.py`) |
| yfinance daily prices | **Migrated** to `io/prices.py` (+ month-end resampling) |
| expanding-window `zscore` | **Migrated** to `features/transforms.py` with anti-leakage tests |
| dashboard, AI brief | **Migrated** — `report/plots.strategy_dashboard`, `report/brief.py` (Claude call + local fallback) |
| yfinance *fundamentals* + synthetic annual→quarterly backfill / back-extrapolation | **Removed** — replaced by WSTS (40y) + Taiwan revenue (20y) + FRED |
| hand-weighted z-score composite | **Replaced** by an estimated dynamic factor (`cycle/dfm.py`) |
| in-sample forward-return averages (n = 1–3 per phase) | **Replaced** by purged walk-forward nowcast + a cost-aware backtest with deflated Sharpe / PBO |
| single scripts | **Replaced** by a package, config, tests, CI |

---

## Disclaimer

Educational and research use only. Not investment advice. Free data is delayed,
revised, and incomplete; results are illustrative.
