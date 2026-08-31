"""`semicycle <command>` — the same steps as the numbered scripts."""

from __future__ import annotations

import argparse

from .pipeline import run_features, run_ingest, run_nowcast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semicycle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="download all sources into DuckDB + parquet")
    sub.add_parser("features", help="build the modelling panel")
    sub.add_parser("nowcast", help="walk-forward nowcast + scoreboard + figure")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
