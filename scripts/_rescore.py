"""Recompute scoreboards, figures and the NOTES_IT results block from the cached
reports/nowcast_oos_h*.csv — without rerunning the slow walk-forward. Run this
after tweaking metrics or plots; run the full `03_nowcast.py` after changing data
or models.
"""

import pandas as pd

from semicycle.config import load_config
from semicycle.nowcast.evaluate import scoreboard
from semicycle.report.plots import nowcast_oos

cfg = load_config()
blocks = []
for h in cfg.params.target.horizons:
    path = cfg.reports_dir / f"nowcast_oos_h{h}.csv"
    if not path.exists():
        continue
    results = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    board = scoreboard(results)
    board.round(4).to_csv(cfg.reports_dir / f"scoreboard_h{h}.csv")
    nowcast_oos(results, h, cfg.reports_dir / f"nowcast_oos_h{h}.png")
    span = f"{results.index.min():%Y-%m} … {results.index.max():%Y-%m}"
    print(f"\n=== h={h} | {len(results)} OOS months ({span}) ===")
    print(board.round(4).to_string())
    blocks.append(f"**h = {h} months** — {len(results)} OOS months ({span})\n\n"
                  + board.round(4).to_markdown() + "\n")

notes = (cfg.root / "NOTES_IT.md").read_text(encoding="utf-8")
start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
if start in notes and end in notes:
    body = "\n".join(blocks)
    notes = notes[: notes.index(start) + len(start)] + "\n" + body + "\n" + notes[notes.index(end):]
    (cfg.root / "NOTES_IT.md").write_text(notes, encoding="utf-8")
    print("\nNOTES_IT.md results block updated.")
