"""Run the Semiconductor Cycle Intelligence System end-to-end."""

from __future__ import annotations

import sys
from pathlib import Path

from ai_brief import generate_brief
from cycle_classifier import classify_series
from cycle_indicators import build_indicators
from dashboard import create_dashboard
from data_collector import collect_all
from forward_returns import compute_forward_returns


def main() -> int:
    print("=" * 60)
    print("Semiconductor Cycle Intelligence System")
    print("=" * 60)

    print("\n[1/5] Collecting market data (yfinance)...")
    data = collect_all()
    revenue, inventory, prices = data["revenue"], data["inventory"], data["prices"]
    print(f"  Revenue quarters: {len(revenue.columns)} tickers, {len(revenue)} periods")
    print(f"  Inventory quarters: {len(inventory.columns)} tickers, {len(inventory)} periods")
    print(f"  Daily prices: {list(prices.columns)}, {len(prices)} days")

    print("\n[2/5] Building cycle indicators...")
    indicators = build_indicators(revenue, inventory, prices)
    if indicators.empty:
        print("ERROR: Could not build indicators — check yfinance data availability.")
        return 1
    print(f"  {len(indicators)} quarterly observations")

    print("\n[3/5] Classifying cycle phases...")
    classified = classify_series(indicators)
    if classified.empty:
        print("ERROR: Could not classify cycle phases — insufficient overlapping data.")
        return 1
    current = classified.iloc[-1]
    print(f"  Current phase: {current['phase']}")
    print(f"  Cycle index: {current['cycle_index']:.2f}")

    print("\n[4/5] Computing forward returns by phase...")
    _, forward_summary = compute_forward_returns(classified, prices)
    if not forward_summary.empty:
        print(forward_summary.to_string(index=False))
    else:
        print("  No forward return summary available.")

    print("\n[5/5] Generating dashboard and AI brief...")
    dashboard_path = create_dashboard(
        classified,
        forward_summary,
        prices=prices,
        data_as_of=data.get("data_as_of"),
    )
    brief = generate_brief(classified, forward_summary)

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Dashboard saved: {dashboard_path.resolve()}")
    print(f"Brief saved:     {Path('semiconductor_cycle_brief.txt').resolve()}")
    print("\n--- AI Brief Preview ---\n")
    preview = brief[:2000] + ("..." if len(brief) > 2000 else "")
    try:
        print(preview)
    except UnicodeEncodeError:
        print(preview.encode("ascii", errors="replace").decode("ascii"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
