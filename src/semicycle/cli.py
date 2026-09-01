"""`semicycle <command>` — the same steps as the numbered scripts."""

from __future__ import annotations

import argparse
import warnings

from .pipeline import (
    run_features,
    run_ingest,
    run_nowcast,
    run_realtime,
    run_report,
    run_strategy,
)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(prog="semicycle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download all sources into DuckDB + parquet")
    sub.add_parser("features", help="build the modelling panel (incl. the cycle factor)")
    sub.add_parser("nowcast", help="walk-forward nowcast + scoreboard + figure")
    sub.add_parser("backtest", help="cycle-timing strategy: stats, deflated Sharpe, PBO")
    sub.add_parser("report", help="regenerate the written sector brief")
    sub.add_parser("realtime", help="nowcast-timing vs factor-timing; recursive-factor check")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        print(run_ingest().to_string(index=False))
    elif args.cmd == "features":
        panel = run_features()
        print(f"panel: {panel.shape[0]} months x {panel.shape[1]} cols "
              f"({panel.index.min().date()}..{panel.index.max().date()})")
    elif args.cmd == "nowcast":
        for h, res in run_nowcast().items():
            print(f"\n=== horizon h={h} | {res['n_oos']} OOS months | "
                  f"{res['n_features']} features ===")
            print(res["scoreboard"].round(4).to_string())
    elif args.cmd == "backtest":
        r = run_strategy()
        print(r["card"])
        print(r["stats"].round(4).to_string())
    elif args.cmd == "report":
        print(run_report()["brief"])
    elif args.cmd == "realtime":
        r = run_realtime()
        cols = ["ann_return", "ann_vol", "sharpe", "max_drawdown"]
        print(f"window {r['window']}")
        print(r["compare"][cols].round(4).to_string())
        print(f"pseudo-real-time vs recursive factor: corr {r['factor_corr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
