"""Module 4 real-time checks.

  1. Time the strategy off the forward nowcast (real-time OOS forecasts) instead
     of the coincident cycle factor, and compare both to buy-and-hold.
  2. Check the pseudo-real-time factor (Kalman-filtered, parameters fixed) against
     a fully recursive one (parameters re-estimated every month).
"""

import warnings

from semicycle.pipeline import run_realtime

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    r = run_realtime()
    print(f"timing the cycle -- coincident factor vs forward nowcast ({r['window']}):\n")
    print(r["compare"][["ann_return", "ann_vol", "sharpe", "max_drawdown"]].round(4).to_string())
    print(f"\npseudo-real-time vs fully recursive factor: corr {r['factor_corr']:.3f}")
    print(f"\nfigures: {', '.join(str(f) for f in r['figures'])}")
