"""A short sector brief from the pipeline outputs.

Migrated from the previous project's `ai_brief.py`: assemble the current cycle
reading, the nowcast, and the strategy stance into a structured prompt, ask
Claude for a brief, and fall back to a deterministic local template when no
`ANTHROPIC_API_KEY` is set (or the call fails).
"""

from __future__ import annotations

import os
import textwrap

import pandas as pd

_MODELS = ("claude-sonnet-4-5", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022")


def _facts(cycle: pd.DataFrame, chronology: pd.DataFrame, scoreboard: pd.DataFrame,
           stats: pd.DataFrame, dsr: pd.Series, pbo: pd.Series, weight_now: float) -> dict:
    last = cycle.iloc[-1]
    span = chronology.iloc[-1]
    return {
        "as_of": cycle.index[-1].strftime("%Y-%m"),
        "phase": last["phase"],
        "phase_since": span["start"].strftime("%Y-%m"),
        "factor": round(float(last["factor"]), 2),
        "regime": last["regime"],
        "nowcast_skill_h3": round(float(scoreboard.loc["lightgbm", "skill_vs_AR"]), 3)
        if "lightgbm" in scoreboard.index else None,
        "strategy_sharpe": round(float(stats.loc["sharpe", "strategy"]), 2),
        "buyhold_sharpe": round(float(stats.loc["sharpe", "buy_hold_SOXX"]), 2),
        "strategy_maxdd": round(float(stats.loc["max_drawdown", "strategy"]), 2),
        "buyhold_maxdd": round(float(stats.loc["max_drawdown", "buy_hold_SOXX"]), 2),
        "deflated_sharpe": round(float(dsr["deflated_sharpe_ratio"]), 2),
        "pbo": round(float(pbo["pbo"]), 2),
        "weight_now": round(float(weight_now), 2),
    }


def _local_brief(f: dict) -> str:
    lean = "above" if f["factor"] > 0 else "below"
    return textwrap.dedent(f"""\
    SEMICONDUCTOR CYCLE BRIEF -- {f['as_of']}

    Cycle position
      The dynamic factor model places the semiconductor cycle in {f['phase']}
      (since {f['phase_since']}), with the latent factor at {f['factor']:+.2f}
      standard deviations -- {lean} its long-run average, {f['regime']}.

    Nowcast
      Out-of-sample, the feature model (with the cycle factor) beats the
      autoregressive benchmark by {f['nowcast_skill_h3']:+.1%} on 3-month-ahead
      billings-growth MAE, and more at turning points.

    Strategy stance
      The cycle-timing overlay currently holds {f['weight_now']:.2f}x SOXX.
      Over 2005-2025 it earned a Sharpe of {f['strategy_sharpe']} against
      {f['buyhold_sharpe']} for buy-and-hold -- i.e. it did NOT add risk-adjusted
      return -- while cutting the maximum drawdown from {f['buyhold_maxdd']:.0%}
      to {f['strategy_maxdd']:.0%}. It is a risk overlay, not an alpha source.

    Robustness
      Deflated Sharpe ratio {f['deflated_sharpe']}, probability of backtest
      overfitting {f['pbo']:.0%} across the 16-config grid. Read the strategy as
      a de-risking rule, and do not go looking for the config that "wins".

    (Local template -- set ANTHROPIC_API_KEY for a written brief.)
    """)


def generate_brief(cycle, chronology, scoreboard, stats, dsr, pbo, weight_now) -> str:
    f = _facts(cycle, chronology, scoreboard, stats, dsr, pbo, weight_now)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _local_brief(f)
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are a semiconductor sector analyst. Write a concise brief (200-300 words, "
            "plain text, no markdown headers) from these pipeline facts. Be measured; the "
            "strategy is a risk overlay, not alpha. Facts:\n"
            + "\n".join(f"- {k}: {v}" for k, v in f.items())
        )
        for model in _MODELS:
            try:
                msg = client.messages.create(
                    model=model, max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception:  # noqa: BLE001, PERF203 - try the next model id
                continue
    except Exception:  # noqa: BLE001
        pass
    return _local_brief(f)
