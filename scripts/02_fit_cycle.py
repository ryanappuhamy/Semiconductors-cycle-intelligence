"""Fit the latent semiconductor cycle and date its turning points.

Writes:
  data/processed/cycle.parquet   — factor, regime, phase, per month
  reports/cycle_factor.png       — factor + phases + turning points + WSTS YoY
  reports/cycle_chronology.csv   — one row per phase span
Prints the loadings, the chronology, and the current cycle position.
"""

import warnings

import pandas as pd

from semicycle.config import load_config
from semicycle.cycle.dating import date_cycle
from semicycle.cycle.dfm import cycle_factor_pit, fit_cycle_factor
from semicycle.report.plots import cycle_chart

warnings.filterwarnings("ignore")


def main() -> int:
    cfg = load_config()

    print("fitting dynamic factor model ...")
    cf = fit_cycle_factor(cfg)
    print("\nindicator loadings:")
    print(cf.summary().round(3).to_string())

    dated = date_cycle(cf.factor, cfg)
    phases, turns, chron = dated["phases"], dated["turns"], dated["chronology"]

    out = cfg.root / "data" / "processed" / "cycle.parquet"
    phases.to_parquet(out)
    chron.to_csv(cfg.reports_dir / "cycle_chronology.csv", index=False)

    print(f"\n{len(turns)} turning points | {len(chron)} phases "
          f"| avg {chron['months'].mean():.0f} months")
    print("\nrecent chronology:")
    show = chron.tail(8).copy()
    show["start"] = show["start"].dt.strftime("%Y-%m")
    show["end"] = show["end"].dt.strftime("%Y-%m")
    print(show.to_string(index=False))

    cur = phases.iloc[-1]
    print(f"\ncurrent: {cur['phase']}  (factor {cur['factor']:+.2f}, {cur['regime']})")

    panel = pd.read_parquet(cfg.panel_path) if cfg.panel_path.exists() else None
    target = panel["target"] if panel is not None else fit_cycle_factor(cfg).factor * 0
    fig = cycle_chart(phases, turns, target, cfg.reports_dir / "cycle_factor.png",
                      pit_factor=cycle_factor_pit(cfg))
    print(f"\nfigure: {fig}")
    print(f"chronology: {cfg.reports_dir / 'cycle_chronology.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
