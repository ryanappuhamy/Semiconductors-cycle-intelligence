# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Real-time data vintages (ALFRED) for a fully point-in-time feature set.
- Cross-sectional strategy (stock-level tilt) alongside the index-timing overlay.

## [0.5.0] — 2026-09-01

Module 4: real-time checks.

### Added
- **Nowcast-based timing** (`strategy.signal.nowcast_signal` /
  `build_weights_nowcast`) — time the overlay off the walk-forward nowcast's
  real-time 3-month-ahead forecast instead of the coincident factor.
- **`cycle.dfm.cycle_factor_recursive`** — fully recursive factor (DFM
  re-estimated every month), to check the pseudo-real-time (params-fixed)
  approximation.
- **`scripts/06_realtime.py`** (`pipeline.run_realtime`) — the timing comparison
  + the factor check; writes `reports/realtime_timing_compare.{png,csv}` and
  `reports/realtime_factor_compare.png`.
- **`report/dashboard.py`** + **`scripts/07_dashboard.py`** — one self-contained
  `reports/dashboard.html` (images embedded as base64), CSP-safe, no external
  requests.
- CLI `semicycle realtime`; `Makefile` `realtime` / `dashboard` targets; `all`
  runs the full chain.
- Tests: `test_realtime.py`. 28 total.

### Results
- On 2009–2025 (the nowcast signal's window — drawdown-light): buy-and-hold
  Sharpe 1.00, **nowcast timing 0.88**, **factor timing 0.83**. Timing off the
  forward nowcast beats timing off the coincident factor (+3 pp/yr return,
  +0.05 Sharpe) — confirming the Module 3 diagnosis that a coincident signal is
  too late — but neither beats buy-and-hold on a window with almost no drawdown.
- Pseudo-real-time factor vs fully recursive factor: correlation **0.94** (over
  2007–, once all six inputs have history). The parameter-fixing shortcut in
  Module 2 does not distort the factor path.

### Changed
- Version 0.4.0 → 0.5.0.

## [0.4.0] — 2026-09-01

Module 3: the cycle-timing strategy and its honesty checks.

### Added
- **`strategy/signal.py`** — point-in-time cycle signal (expanding z-score of the
  DFM factor + its 6-month change) → a long-only target weight on SOXX,
  `clip(base + gain·s, min, max)`.
- **`strategy/backtest.py`** — monthly backtest engine with an explicit no-look-
  ahead timing convention and transaction costs on turnover; `grid_returns()`
  produces the per-config return matrix for the robustness stats.
- **`strategy/stats.py`** — `perf_stats` (Sharpe, Sortino, max drawdown, Calmar,
  skew/kurtosis); **`deflated_sharpe_ratio`** (Bailey & López de Prado 2014);
  **`probability_of_backtest_overfitting`** (CSCV; Bailey, Borwein, López de
  Prado, Zhu 2017); `regime_attribution` by the Module 2 cycle phase.
- **`scripts/04_backtest.py`**, **`scripts/05_report.py`**;
  `report.plots.equity_curve` + `strategy_dashboard`; **`report/brief.py`**
  (migrated `ai_brief.py` — structured prompt → Claude, local template fallback).
- `config/params.yaml` → `strategy:` (asset, signal weights, 16-config grid,
  CSCV partitions, backtest window).
- CLI: `semicycle backtest`, `semicycle report`. `Makefile`: `all` runs the
  full chain; `backtest` / `report` are first-class targets.
- Tests: `test_strategy.py` (backtest mechanics, deflated Sharpe, PBO edge
  cases). 25 total.

### Results
- 2005–2025, 10 bps cost, vs buy-and-hold SOXX: strategy Sharpe **0.59** vs
  **0.70**; annualised return +10.9% vs +15.5%; vol 21.6% vs 25.0%; max drawdown
  −54% vs −60%. The overlay cuts risk but not below buy-and-hold's Sharpe — the
  ~0.1 gap is stable across assets (SMH, equal-weight basket) and sub-periods.
  A **risk overlay, not alpha**: semiconductor equities price the fundamental
  cycle before it reaches billings.
- By phase: +23% / +20% annualised in Early / Late Cycle, −8% in Downturn.
- Deflated Sharpe ratio 0.99 (P[true SR > 0] over N = 16 configs); PBO 0.32
  (CSCV, 252 splits).

### Fixed
- WSTS / price data: backtest capped at 2025-12 — yfinance semiconductor prices
  show implausible +30–40% monthly moves from 2026-04 (ETFs moving 2× their top
  holding), the same era as the corrupt WSTS 2026 rows.

### Changed
- Version 0.3.0 → 0.4.0.

## [0.3.0] — 2026-09-01

Module 2: the latent semiconductor cycle.

### Added
- **`cycle/dfm.py`** — dynamic factor model (`statsmodels DynamicFactorMQ`,
  1 factor, AR(2), idiosyncratic AR(1), EM). Replaces v1's hand-weighted
  z-score index with an estimated common component. Inputs (`config/params.yaml`
  → `cycle:`): four WSTS regional billings YoY + Taiwan value-chain revenue YoY
  + SOX 12-month momentum. `fit_cycle_factor()` returns the full-sample smoothed
  factor (for the chronology); `cycle_factor_pit()` returns the pseudo-real-time
  Kalman-filtered factor on `as_of_panel` inputs (for the nowcast feature).
- **`cycle/dating.py`** — Bry–Boschan turning-point dating (`bry_boschan`) and
  the four-phase map (`classify_phases`, `phase_chronology`, `date_cycle`):
  alternating local extrema, ≥5-month phases, ≥18-month cycles, edge censoring;
  Early/Mid/Late/Downturn from the turns + the factor's zero line, with a
  minimum-run filter. Rules in `config/params.yaml` → `cycle.dating`.
- **`scripts/02_fit_cycle.py`** — fits + dates the cycle, writes
  `data/processed/cycle.parquet`, `reports/cycle_chronology.csv`,
  `reports/cycle_factor.png` (`report.plots.cycle_chart`).
- Cycle factor added to the modelling panel as `cycle_factor`,
  `cycle_factor__chg3`, `cycle_factor__chg6` (`features.build._add_cycle_factor`).
- Tests: `test_cycle_dfm.py`, `test_cycle_dating.py` (17 total).

### Results
- Fitted factor correlates 0.95 with WSTS 3MMA-YoY; 22 turning points, 35 phases
  over 1987–2026 (~14-month average phase). Matches the record: 2000 peak,
  2001 bust, GFC, 2019 memory glut, 2021 shortage boom, 2023 trough. Current
  reading: Mid Cycle, factor +1.7.
- Adding the factor to the nowcast flips the feature models past the AR
  benchmark: LightGBM skill vs AR goes −9.3% → **+3.5%** at h = 3 and
  −0.6% → **+6.7%** at h = 6; turning-point MAE −11 to −15% vs AR.
  `cycle_factor__chg6` is the top LightGBM feature at h = 6.

### Changed
- `Makefile`: `all` now runs `ingest features cycle nowcast`; `cycle` is a
  first-class target.

## [0.2.0] — 2026-08-31

Ground-up rebuild from a small didactic toolkit into a reproducible research
pipeline. The economic intuition (3 signal pillars, the 4-phase taxonomy) is
kept; the data foundation and all the statistics are replaced. v1 is preserved
at tag [`v1`](https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v1).

### Added
- `semicycle` package under `src/`, config-driven (`config/sources.yaml`,
  `config/params.yaml`), `Makefile`, GitHub Actions CI, 10 tests, `ruff` clean.
- **Data loaders** (`io/`): WSTS Blue Book billings (1986–), FinMind Taiwan
  monthly revenue (2005–), FRED macro (no API key), yfinance prices → tidy
  `(date, series, value, published)` in a DuckDB store; `data/raw/MANIFEST.json`.
- **Point-in-time feature construction** (`features/as_of_panel`): each feature
  at month `T` uses only data whose publication date was `≤ T`, from per-source
  `release_lag_days`. Causal stationary transforms (YoY, moving averages,
  expanding z-scores). Anti-leakage tests.
- **Nowcast** (`nowcast/`): supervised panel; an autoregressive point-in-time
  benchmark plus ElasticNet and LightGBM; expanding-window walk-forward CV with
  purge + embargo (López de Prado); full-sample and turning-point-conditional
  metrics; full-sample LightGBM feature importances.
- **Reports**: committed scoreboards (`reports/scoreboard_h*.csv`), out-of-sample
  figures (`reports/nowcast_oos_h*.png`), feature importances.
- `NOTES_IT.md` lab notebook (Italian); `README.md` rebuilt.
- WSTS loader guards against a corrupt release: caps actuals at a trusted month
  and warns on implausible month-on-month moves (the Jun-2026 book's 2026 rows
  imply ~2× the real market size).

### Changed
- Cycle index: hand-weighted average of three z-scores → (Module 2) an estimated
  dynamic factor. Interim: the nowcast target is WSTS worldwide 3MMA YoY.
- Validation: in-sample forward-return averages by phase (n = 1–3 per cell) →
  240-month purged/embargoed walk-forward against an honest benchmark.
- Prices and the expanding z-score migrated from the v1 scripts into the package
  with tests.

### Removed
- yfinance quarterly *fundamentals* and the synthetic annual→quarterly backfill /
  back-extrapolation — replaced by WSTS (40y) + Taiwan revenue (20y) + FRED.
- v1 single-file scripts (`main.py`, `data_collector.py`, `cycle_indicators.py`,
  `cycle_classifier.py`, `forward_returns.py`, `dashboard.py`, `ai_brief.py`) —
  preserved in history and at tag `v1`; `dashboard.py` / `ai_brief.py` return in
  Module 3.

### Results (Module 1)
- 240 out-of-sample months (2006–2025). On full-sample MAE the point-in-time
  autoregression wins at every horizon (the 3MMA-YoY target is ~0.95
  autocorrelated). The feature models' edge is at inflections: LightGBM cuts
  turning-point MAE by 3.5% at h = 3 and 5.6% at h = 6 and calls the direction
  of change more often. Taiwan revenue adds timeliness, not orthogonal
  information (WSTS "Asia Pacific" already carries it).

## [0.1.0] — earlier

Original **Semiconductor Cycle Intelligence System**: a Python toolkit that built
a semiconductor cycle index from three z-scored signals (equipment revenue
growth, inventory-to-revenue, SOXX-vs-QQQ momentum), classified each quarter into
one of four cycle phases, and produced a matplotlib dashboard plus an
AI-generated sector brief. All data from yfinance. Limited by ~5 usable quarterly
observations after alignment.

[Unreleased]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v0.2.0
[0.1.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v1
