"""Assemble data/processed/panel.parquet from the raw tables."""

from semicycle.pipeline import run_features

if __name__ == "__main__":
    panel = run_features()
    cov = panel.notna().mean().sort_values()
    print(f"panel: {panel.shape[0]} months x {panel.shape[1]} columns")
    print(f"range: {panel.index.min().date()} .. {panel.index.max().date()}")
    print(f"target coverage: {panel['target'].notna().mean():.1%}")
    print("\nleast-covered columns:")
    print(cov.head(8).round(3).to_string())
