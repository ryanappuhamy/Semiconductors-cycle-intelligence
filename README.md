# Semiconductor Cycle Intelligence System

A Python research toolkit that tracks the semiconductor industry cycle using public market data. The system combines fundamental signals (equipment spending and inventory), relative equity momentum, and historical forward-return analysis into a single composite index — then classifies the current environment into one of four cycle phases and produces a visual dashboard plus an AI-generated sector brief.

---

## Methodology Overview

Semiconductor demand moves in multi-year cycles driven by capacity investment, inventory build/drawdown, and end-market demand. This project approximates that cycle by monitoring three complementary signals that tend to lead or confirm sector turning points:

1. **Supply-side investment** — when equipment makers grow revenue, fabs are expanding capacity (early/mid cycle).
2. **Inventory pressure** — when inventory builds relative to sales, the sector is often oversupplied (late cycle / downturn).
3. **Market pricing** — when semiconductor equities outperform the broad market, risk appetite and expectations are elevated.

Each signal is normalized to a z-score, combined into a **Semiconductor Cycle Index**, and mapped to a phase label. Historical phase assignments are then used to estimate average forward returns for selected semiconductor stocks.

---

## The Three Indicators

| Indicator | Construction | Weight | Rationale |
|-----------|--------------|--------|-----------|
| **Equipment Revenue Growth** | Average revenue growth across ASML, AMAT, and LRCX. Uses year-over-year growth when ≥8 quarters of history are available; otherwise quarter-over-quarter. | 40% | Equipment makers are the first link in the supply chain. Rising revenue reflects fab capex and capacity expansion — a hallmark of early and mid-cycle recovery. |
| **Inventory-to-Revenue Ratio** | Average of inventory ÷ revenue for MU and TXN. The ratio is **inverted** before z-scoring so that high inventory (late-cycle signal) scores negatively. | 30% | Memory and analog semis are sensitive to channel inventory. Elevated inventory relative to sales often precedes margin compression and destocking phases. |
| **SOXX vs QQQ Relative Momentum** | 6-month rolling total return of SOXX minus QQQ, sampled at quarter-end dates. | 30% | Captures how the market is pricing semiconductor risk relative to large-cap tech. Persistent outperformance can indicate late-cycle exuberance; underperformance often coincides with downturns. |

All three indicators are z-scored using an expanding window (minimum 3 observations) so readings are interpreted relative to their own history rather than absolute levels.

---

## Cycle Phases

The **Semiconductor Cycle Index** is a weighted average of the three z-scored indicators. Each quarter is classified using the index **level** and its **quarter-over-quarter change** (direction):

| Phase | Index Level | Direction | Interpretation |
|-------|-------------|-----------|----------------|
| **Early Cycle** | Below 0 | Improving (≥ 0) | Recovery from trough; fundamentals stabilizing or inflecting upward. |
| **Mid Cycle** | At or above 0 | Improving (≥ 0) | Expansion phase; capex and demand growing in tandem. |
| **Late Cycle** | At or above 0 | Deteriorating (< 0) | Peak conditions; growth slowing while sentiment may still be elevated. |
| **Downturn** | Below 0 | Deteriorating (< 0) | Contraction; falling demand, inventory corrections, and weak pricing. |

The first observed quarter is labeled from its level alone (Mid Cycle if ≥ 0, otherwise Early Cycle) since no prior direction is available.

---

## Project Structure

```
├── main.py                 # End-to-end pipeline entry point
├── data_collector.py       # yfinance data download (fundamentals + prices)
├── cycle_indicators.py     # Indicator construction and z-score normalization
├── cycle_classifier.py     # Composite index and phase classification
├── forward_returns.py      # Historical forward returns by phase (NVDA, MU, MRVL)
├── dashboard.py            # Multi-panel matplotlib/seaborn visualization
├── ai_brief.py             # Claude API sector brief generation
└── requirements.txt
```

---

## Data Sources and Limitations

All data is sourced from **Yahoo Finance** via the [`yfinance`](https://github.com/ranaroussi/yfinance) library. No paid APIs are required for market data.

### Fundamentals (quarterly)

| Data | Tickers |
|------|---------|
| Revenue | ASML, AMAT, LRCX, MU, TXN, TSM |
| Inventory | MU, TXN |

### Prices (daily, 10-year lookback)

| Data | Tickers |
|------|---------|
| Semiconductor equities & ETF | NVDA, MU, MRVL, VRT, SOXX |
| Broad tech benchmark | QQQ |

### Known limitations

- **~6 quarters of quarterly financials.** Yahoo Finance exposes only a short history of quarterly income statements and balance sheets per ticker. After alignment, year-over-year equipment growth, and z-score warm-up, the system typically produces **3–5 usable quarterly observations**. Forward-return statistics by phase will be sparse until more history accumulates.
- **Fiscal calendar mismatches.** Report dates are normalized to calendar quarter-ends, but companies operate on different fiscal calendars; some quarters may have partial ticker coverage.
- **Free data quality.** yfinance data can be delayed, revised, or incomplete. Results should be treated as illustrative, not audit-grade.
- **AI brief requires an API key.** The optional Claude brief needs a valid `ANTHROPIC_API_KEY`. The primary model (`claude-sonnet-4-20250514`) may no longer be available; the system automatically falls back to newer Sonnet models.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone <repository-url>
cd "Backtest strategia"
pip install -r requirements.txt
```

### Dependencies

- `yfinance` — market data
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualization
- `anthropic` — AI brief generation (optional)

---

## Usage

Run the full pipeline:

```bash
python main.py
```

To enable the AI-generated sector brief, set your Anthropic API key first:

```bash
# PowerShell
$env:ANTHROPIC_API_KEY = "your-api-key-here"
python main.py
```

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="your-api-key-here"
python main.py
```

### Pipeline steps

1. Download quarterly revenue/inventory and daily prices
2. Build and z-score the three cycle indicators
3. Compute the composite index and classify each quarter
4. Calculate average 3-, 6-, and 12-month forward returns by phase for NVDA, MU, and MRVL
5. Save the dashboard image and AI brief

### Outputs

| File | Description |
|------|-------------|
| `semiconductor_cycle_dashboard.png` | Four-panel chart: cycle index with phase shading, current indicator levels, phase classification card, forward-return heatmap |
| `semiconductor_cycle_brief.txt` | Structured sector brief (Claude-generated, or local fallback if no API key) |

---

## Example Output

Console summary from a recent run:

```
[3/5] Classifying cycle phases...
  Current phase: Late Cycle
  Cycle index: 0.40

[4/5] Computing forward returns by phase...
     phase ticker  horizon_months  avg_forward_return  n_obs
Late Cycle   MRVL               3            0.166407      1
Late Cycle     MU               3            0.184198      1
Late Cycle   NVDA               3           -0.064829      1
```

Excerpt from the AI brief (`semiconductor_cycle_brief.txt`):

> **Late Cycle** conditions persist as of Q1 2026, with cycle index at **0.403** – down sharply from 0.859 in Q4 2025. The **-0.456 QoQ decline** signals meaningful late-cycle deterioration.
>
> - **Equipment Revenue Growth: -0.838σ** — below-trend capex spending
> - **Inventory-to-Revenue: +1.37σ** — elevated inventory suggests destocking ahead
> - **SOXX Relative Momentum: +1.09σ** — equities still outperforming despite weakening fundamentals

The dashboard overlays phase-colored bands on the composite index time series, bar charts of current z-scores, and a heatmap of historical forward returns by phase.

---

## Disclaimer

**This project is for educational and research purposes only. It does not constitute financial advice, investment recommendations, or an offer to buy or sell any securities.**

Past forward-return statistics do not guarantee future performance. The model relies on limited, freely available data and simplified heuristics that may not capture the full complexity of semiconductor industry dynamics. Always conduct your own due diligence and consult a qualified financial advisor before making investment decisions.
