"""Assemble reports/dashboard.html — one self-contained page for the whole pipeline."""

import warnings

from semicycle.config import load_config
from semicycle.cycle.dating import date_cycle
from semicycle.cycle.dfm import fit_cycle_factor
from semicycle.report.dashboard import build_dashboard

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cfg = load_config()
    chron = date_cycle(fit_cycle_factor(cfg).factor, cfg)["chronology"]
    brief_path = cfg.reports_dir / "semiconductor_cycle_brief.txt"
    brief = brief_path.read_text(encoding="utf-8") if brief_path.exists() else ""
    out = build_dashboard(cfg, brief=brief, chronology=chron)
    print(f"dashboard: {out}  ({out.stat().st_size // 1024} KB)")
