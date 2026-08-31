"""Download every source into data/raw/*.parquet and the DuckDB store.

Sources: WSTS billings, Taiwan monthly revenue (FinMind), equity prices
(yfinance), FRED macro (best-effort). Writes data/raw/MANIFEST.json.
"""

from semicycle.pipeline import run_ingest

if __name__ == "__main__":
    manifest = run_ingest()
    print("\nMANIFEST")
    print(manifest.to_string(index=False))
