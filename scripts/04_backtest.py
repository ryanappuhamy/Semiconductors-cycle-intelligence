"""Backtest the cycle-timing strategy with costs, deflated Sharpe and PBO."""

import warnings

from semicycle.pipeline import run_strategy

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    r = run_strategy()
    print(r["card"])
    print("performance:")
    print(r["stats"].round(4).to_string())
    print("\nper cycle phase (strategy):")
    print(r["regime"].round(4).to_string())
    print(f"\nfigures: {', '.join(str(f) for f in r['figures'])}")
