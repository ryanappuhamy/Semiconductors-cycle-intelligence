# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- **Module 2 — cycle factor.** `cycle/dfm.py`: mixed-frequency dynamic factor
  model (`statsmodels DynamicFactorMQ`) over WSTS billings, Taiwan revenue, FRED
  activity and equity momentum → one latent coincident cycle index.
  `cycle/dating.py`: Bry–Boschan turning-point dating → Early/Mid/Late/Downturn
  chronology. Factor fed back as a nowcast feature.
- **Module 3 — strategy.** Cycle-driven equity strategy on `^SOX`/`SMH` + a
  cross-sectional tilt; monthly rebalance with explicit costs; deflated Sharpe
  ratio, probability of backtest overfitting (CSCV), turnover, per-regime stats.
  Migrates the v1 dashboard and written brief.
- **Module 4.** Real-time data vintages (ALFRED), static HTML dashboard, final
  write-up.

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

[Unreleased]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v0.2.0
[0.1.0]: https://github.com/ryanappuhamy/Semiconductors-cycle-intelligence/releases/tag/v1
