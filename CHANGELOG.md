# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- **Module 3 — strategy.** Cycle-driven equity strategy on `^SOX`/`SMH` + a
  cross-sectional tilt; monthly rebalance with explicit costs; deflated Sharpe
  ratio, probability of backtest overfitting (CSCV), turnover, per-regime stats.
  Migrates the v1 dashboard and written brief.
- **Module 4.** Fully recursive real-time factor, data vintages (ALFRED), static
  HTML dashboard, final write-up.

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

[Unreleased]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v0.2.0
[0.1.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v1
