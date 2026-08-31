"""Walk-forward nowcast of the semiconductor cycle: scoreboard + OOS figure."""

from semicycle.pipeline import run_nowcast

if __name__ == "__main__":
    for h, res in run_nowcast().items():
        print(f"\n=== horizon h={h} months | {res['n_oos']} out-of-sample months "
              f"| {res['n_features']} features ===")
        print(res["scoreboard"].round(4).to_string())
        print("\ntop features (LightGBM gain, full-sample, interpretation only):")
        cols = ["feature", "gain_share", "family"]
        print(res["importance"].head(10)[cols].round(3).to_string(index=False))
        print(f"figure: {res['figure']}")
