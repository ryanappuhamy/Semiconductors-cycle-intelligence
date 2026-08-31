# Semiconductor Cycle Intelligence — one-command pipeline.
# Windows: `make` via Git Bash / WSL, or run the python commands directly.

PY := python

.PHONY: help install ingest features cycle nowcast backtest report test lint all clean

help:
	@echo "targets: install | ingest | features | cycle | nowcast | test | lint | all"
	@echo "  (backtest | report are stubs, landing in Module 3)"

install:
	$(PY) -m pip install -e ".[dev]"

ingest:
	$(PY) scripts/00_ingest.py

features:
	$(PY) scripts/01_build_features.py

cycle:
	$(PY) scripts/02_fit_cycle.py

nowcast:
	$(PY) scripts/03_nowcast.py

backtest:
	$(PY) scripts/04_backtest.py

report:
	$(PY) scripts/05_report.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

all: ingest features cycle nowcast

clean:
	rm -rf data/raw/* data/interim/* data/processed/* reports/*.png
	find . -name '.gitkeep' -exec touch {} +
