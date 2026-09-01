"""Assemble the sector brief (and refresh the strategy dashboard).

Migrates the previous project's ai_brief.py: structured prompt -> Claude, with a
deterministic local template when ANTHROPIC_API_KEY is not set.
"""

import warnings

from semicycle.pipeline import run_report

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    r = run_report()
    print(r["brief"])
    print(f"\nbrief:     {r['path']}")
    print(f"dashboard: {r['strategy']['figures'][1]}")
