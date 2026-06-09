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
  Current phase: Mid Cycle
  Cycle index: 0.74

[4/5] Computing forward returns by phase...
      phase ticker  horizon_months  avg_forward_return  n_obs
   Downturn   MRVL               3           -0.005608     10
   Downturn   MRVL               6            0.107061     10
   Downturn   MRVL              12            0.346217     10
   Downturn     MU               3            0.032374     10
   Downturn     MU               6            0.155046     10
   Downturn     MU              12            0.584970     10
   Downturn   NVDA               3            0.108105     10
   Downturn   NVDA               6            0.383845     10
   Downturn   NVDA              12            1.020781     10
Early Cycle   MRVL               3            0.136133     11
Early Cycle   MRVL               6            0.165145     11
Early Cycle   MRVL              12            0.408205     11
Early Cycle     MU               3            0.101783     11
Early Cycle     MU               6            0.179151     11
Early Cycle     MU              12            0.525741     11
Early Cycle   NVDA               3            0.308011     11
Early Cycle   NVDA               6            0.584448     11
Early Cycle   NVDA              12            1.280069     11
 Late Cycle   MRVL               3            0.241944      2
 Late Cycle   MRVL               6            0.129257      2
 Late Cycle   MRVL              12           -0.285709      1
 Late Cycle     MU               3            0.536747      2
 Late Cycle     MU               6            0.521242      2
 Late Cycle     MU              12           -0.291064      1
 Late Cycle   NVDA               3            0.216774      2
 Late Cycle   NVDA               6            0.111194      2
 Late Cycle   NVDA              12           -0.413558      1
  Mid Cycle   MRVL               3            0.096171      3
  Mid Cycle   MRVL               6            0.312325      2
  Mid Cycle   MRVL              12           -0.251025      1
  Mid Cycle     MU               3            0.126488      3
  Mid Cycle     MU               6            0.742678      2
  Mid Cycle     MU              12           -0.346999      1
  Mid Cycle   NVDA               3            0.050684      3
  Mid Cycle   NVDA               6            0.333364      2
  Mid Cycle   NVDA              12           -0.241629      1
```

Current indicator readings (Q4 2025):

| Metric | Value |
|--------|-------|
| Phase | Mid Cycle |
| Cycle index | 0.74 |
| Direction (QoQ) | +0.94 |
| Equipment revenue growth | +2.12σ |
| Inventory / revenue | −1.19σ (lean) |
| SOXX vs QQQ momentum | +0.85σ |

The dashboard overlays phase-colored bands on the composite index, individual indicator z-scores, log-scale indexed prices for NVDA/MU/MRVL, and a forward-return heatmap across all four phases.

---

## Results and Interpretation

*Based on the most recent pipeline run. Figures will update each time `main.py` is executed.*

### Current Cycle Assessment

The model currently classifies the semiconductor sector in **Mid Cycle**, with a composite index of **0.74** and a positive quarter-over-quarter trajectory. Under the hood, the signal mix is uneven:

| Indicator | Z-Score | Reading |
|-----------|---------|---------|
| Equipment revenue growth (ASML, AMAT, LRCX) | **+2.12σ** | Capex and fab equipment demand are running well above the historical norm |
| Inventory / revenue (MU, TXN) | **−1.19σ** | Channel inventory is lean relative to sales — a constructive setup, not a destocking overhang |
| SOXX vs QQQ momentum | **+0.85σ** | Semiconductor equities continue to outperform broad large-cap tech |

Taken together, this is a **late-stage expansion profile**: fundamentals are still firm and inventories are not bloated, but equipment spending is at an extreme relative to its own history. That combination supports a Mid Cycle label on index level and direction, while flagging that the capex pillar of the model is stretched.

### What This Means for NVDA, MU, and MRVL

Historical forward returns in the dashboard heatmap are conditional on phase — they describe what these names have *averaged* after quarters classified the same way, not a forecast.

**Mid Cycle (current phase)** — The sample here is thin (see below). Over 3 months, historical averages are modestly positive: NVDA ~+5%, MU ~+13%, MRVL ~+10%. At 6 months, MU shows the strongest historical follow-through (~+74%), while NVDA and MRVL sit nearer +31–33%. Twelve-month Mid Cycle averages turn negative in the current dataset, but that row rests on a single observation per name and should not be weighted in positioning decisions.

**Early Cycle** — With a much larger sample (n = 10 per ticker at 3/6/12 months), recoveries have been the most rewarding phase historically: NVDA averaged roughly +31% at 3 months and **+120%** at 12 months; MU and MRVL posted solid double-digit gains at shorter horizons and **+39–57%** at 12 months. Names tend to re-rate aggressively when the model transitions from below-zero index levels back into expansion.

**Downturn** — Counter-intuitively, average forward returns during classified downturns in this backtest are also positive (NVDA ~+102% at 12 months, n = 12). That largely reflects buying recoveries *from* cycle troughs rather than smooth returns *through* contraction. Downturn labels mark stress in the model inputs, not necessarily the moment of maximum equity downside.

**Practical takeaway:** At today's Mid Cycle reading, the heatmap does not point to the same magnitude of upside that Early Cycle episodes delivered historically. Lean inventory is a tailwind, but the extreme equipment growth reading argues against extrapolating mid-cycle returns indefinitely — particularly for names already pricing in a strong capex and AI spending narrative (NVDA) versus memory-levered names that may respond differently to the inventory signal (MU, MRVL).

### Statistical Limitations

Phase-level forward returns are only as reliable as the number of historical quarters assigned to each phase. In the current run:

| Phase | Typical n per cell | Comment |
|-------|-------------------|---------|
| **Early Cycle** | 10–13 | Reasonable for directional comparison across horizons |
| **Downturn** | 10 | Similarly usable, with the caveat that trough-buying dominates long-horizon averages |
| **Mid Cycle** | **1–3** | Insufficient for robust inference; 12-month cells are effectively anecdotal |

Conclusions drawn from Mid Cycle cells should be treated as illustrative. Early Cycle and Downturn statistics carry more weight in this dataset, but still reflect a limited fundamental history (~40 indicator quarters after annual backfill) and a single regime of AI-driven semiconductor demand that may not repeat.

### Key Divergence: Extreme Equipment Growth

The standout tension in the current reading is **equipment revenue growth at +2.12σ** — near the top of the model's observed range — coexisting with a Mid Cycle classification driven by a still-rising composite index. Historically, capex intensity at these levels has tended to mean-revert: fabs slow orders when capacity catches up with demand, equipment maker revenue growth decelerates, and the cycle index rolls from Mid toward Late Cycle or Downturn.

Lean inventory (−1.19σ) mitigates near-term margin risk but does not eliminate capex cyclicality. If equipment growth normalizes while SOXX momentum remains elevated (+0.85σ), the model's typical Late Cycle pattern — positive index level, deteriorating direction — becomes more likely. That is the scenario where historical heatmaps suggest less reliable follow-through and where mean-reversion in the equipment signal is the primary risk to monitor.

---

## Disclaimer

**This project is for educational and research purposes only. It does not constitute financial advice, investment recommendations, or an offer to buy or sell any securities.**

Past forward-return statistics do not guarantee future performance. The model relies on limited, freely available data and simplified heuristics that may not capture the full complexity of semiconductor industry dynamics. Always conduct your own due diligence and consult a qualified financial advisor before making investment decisions.
