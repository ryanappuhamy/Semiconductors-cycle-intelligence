"""Emit reports/dashboard_artifact.html — the body-only HTML for the Artifact tool.

Same content as scripts/07_dashboard.py but as a fragment (no <!doctype>/<head>/<body>)
with a designed stylesheet, for publishing via Claude Code's Artifact tool. Run
after 04_backtest / 05_report / 06_realtime.
"""
# ruff: noqa: E501  (HTML template lines)
import base64
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from semicycle.config import load_config  # noqa: E402
from semicycle.cycle.dating import date_cycle  # noqa: E402
from semicycle.cycle.dfm import fit_cycle_factor  # noqa: E402

cfg = load_config()
R = cfg.reports_dir


def img(name, alt):
    p = R / name
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<figure class="plot"><img src="data:image/png;base64,{b}" alt="{alt}"></figure>'


def tbl(name, drop=None):
    p = R / name
    if not p.exists():
        return ""
    df = pd.read_csv(p, index_col=0)
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.map(lambda v: f"{v:,.3f}" if isinstance(v, float) else v)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for i, r in df.iterrows():
        rows += "<tr><th>" + str(i) + "</th>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
    return f'<div class="tw"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


d = date_cycle(fit_cycle_factor(cfg).factor, cfg)
ph = d["phases"].iloc[-1]
since = d["chronology"].iloc[-1]["start"].strftime("%b %Y")
sb = pd.read_csv(R / "scoreboard_h3.csv", index_col=0)
st = pd.read_csv(R / "strategy_stats.csv", index_col=0)
rob = pd.read_csv(R / "strategy_robustness.csv", index_col=0)["value"]
rt = pd.read_csv(R / "realtime_timing_compare.csv", index_col=0)

phase = ph["phase"]
sev = {"Early Cycle": "good", "Mid Cycle": "good", "Late Cycle": "warn", "Downturn": "crit"}[phase]
sk = sb.loc["lightgbm", "skill_vs_AR"]
chron_tbl = d["chronology"].tail(8).assign(
    start=lambda x: pd.to_datetime(x["start"]).dt.strftime("%Y-%m"),
    end=lambda x: pd.to_datetime(x["end"]).dt.strftime("%Y-%m"),
)
chron_rows = "".join(
    f"<tr><td>{r.phase}</td><td>{r.start}</td><td>{r.end}</td><td>{r.months}</td></tr>"
    for r in chron_tbl.itertuples()
)
brief = (R / "semiconductor_cycle_brief.txt").read_text(encoding="utf-8").strip()

HTML = f"""<title>Silicon Cycle Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {{
  --bg:#f5f6f8; --surface:#ffffff; --sunk:#f0f2f5;
  --ink:#161920; --muted:#5f6672; --line:#e3e5ea;
  --accent:#2f6fed; --accent-soft:#e8effd;
  --good:#17805a; --warn:#b7791f; --crit:#c2413a;
  --good-bg:#e4f2ec; --warn-bg:#f6ecda; --crit-bg:#f7e3e1;
  --shadow:0 1px 2px rgba(20,25,40,.06),0 8px 24px rgba(20,25,40,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0b0e13; --surface:#141922; --sunk:#0f141b;
    --ink:#e5e8ef; --muted:#8b93a2; --line:#232b36;
    --accent:#5b8cff; --accent-soft:#18233b;
    --good:#38b389; --warn:#d69a3f; --crit:#e0655c;
    --good-bg:#122a22; --warn-bg:#2b2415; --crit-bg:#2c1a18;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0b0e13; --surface:#141922; --sunk:#0f141b;
  --ink:#e5e8ef; --muted:#8b93a2; --line:#232b36;
  --accent:#5b8cff; --accent-soft:#18233b;
  --good:#38b389; --warn:#d69a3f; --crit:#e0655c;
  --good-bg:#122a22; --warn-bg:#2b2415; --crit-bg:#2c1a18;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font:400 15px/1.6 "IBM Plex Sans",-apple-system,Segoe UI,Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:940px; margin:0 auto; padding:56px 24px 96px; }}
.eyebrow {{
  font:500 12px/1 "IBM Plex Mono",monospace; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted);
}}
h1 {{ font:700 30px/1.15 "IBM Plex Sans"; margin:10px 0 8px; text-wrap:balance; letter-spacing:-.01em; }}
.lede {{ color:var(--muted); max-width:62ch; margin:0; }}
.strip {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px;
  margin:36px 0 8px;
}}
.tile {{
  background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; box-shadow:var(--shadow);
}}
.tile .k {{ font:500 11px/1 "IBM Plex Mono",monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
.tile .v {{ font:600 24px/1.1 "IBM Plex Sans"; margin-top:10px; font-variant-numeric:tabular-nums; letter-spacing:-.01em; }}
.tile .n {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
.pill {{
  display:inline-block; padding:3px 10px; border-radius:999px; font:600 13px/1.4 "IBM Plex Sans";
}}
.pill.good {{ background:var(--good-bg); color:var(--good); }}
.pill.warn {{ background:var(--warn-bg); color:var(--warn); }}
.pill.crit {{ background:var(--crit-bg); color:var(--crit); }}
section {{ margin-top:52px; }}
section > .eyebrow {{ display:block; }}
h2 {{ font:600 20px/1.3 "IBM Plex Sans"; margin:6px 0 6px; letter-spacing:-.01em; }}
.note {{ color:var(--muted); max-width:64ch; margin:0 0 18px; font-size:14.5px; }}
.plot {{
  margin:0; background:#fff; border:1px solid var(--line); border-radius:12px;
  padding:12px; box-shadow:var(--shadow); overflow:hidden;
}}
.plot img {{ display:block; width:100%; height:auto; border-radius:4px; }}
.tw {{ overflow-x:auto; margin-top:16px; }}
table {{ border-collapse:collapse; width:100%; font:400 13px/1.5 "IBM Plex Mono",monospace; }}
th, td {{ border-bottom:1px solid var(--line); padding:7px 12px; text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
thead th {{ color:var(--muted); font-weight:500; border-bottom:1.5px solid var(--line); }}
tbody th {{ text-align:left; font-weight:500; color:var(--ink); }}
td:first-child, th:first-child {{ text-align:left; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media (max-width:680px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.brief {{
  margin-top:14px; background:var(--sunk); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:8px;
  padding:20px 22px; font:400 13px/1.7 "IBM Plex Mono",monospace;
  white-space:pre-wrap; color:var(--ink);
}}
footer {{ margin-top:64px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:12.5px; }}
a {{ color:var(--accent); }}
</style>

<div class="wrap">
  <span class="eyebrow">Semiconductor cycle intelligence &nbsp;/&nbsp; free public data</span>
  <h1>Silicon Cycle Monitor</h1>
  <p class="lede">A latent cycle factor estimated from worldwide billings, Taiwan
  revenue and equity momentum &mdash; then a walk-forward billings nowcast, then a
  cost-aware timing overlay, each checked out-of-sample.</p>

  <div class="strip">
    <div class="tile">
      <div class="k">Cycle phase</div>
      <div class="v"><span class="pill {sev}">{phase}</span></div>
      <div class="n">since {since} &middot; {ph['regime']}</div>
    </div>
    <div class="tile">
      <div class="k">Latent factor</div>
      <div class="v">{ph['factor']:+.2f}&#8202;&sigma;</div>
      <div class="n">vs its 1987&ndash; average</div>
    </div>
    <div class="tile">
      <div class="k">Nowcast edge (h=3)</div>
      <div class="v">{sk:+.1%}</div>
      <div class="n">MAE skill vs AR benchmark; &minus;15% at turns</div>
    </div>
    <div class="tile">
      <div class="k">Strategy verdict</div>
      <div class="v">risk overlay</div>
      <div class="n">Sharpe {st.loc['sharpe','strategy']:.2f} vs {st.loc['sharpe','buy_hold_SOXX']:.2f} buy &amp; hold</div>
    </div>
  </div>

  <section>
    <span class="eyebrow">01 &nbsp;/&nbsp; The cycle</span>
    <h2>A dynamic factor model, not a hand-weighted index</h2>
    <p class="note">One latent AR(2) factor drives four WSTS regional billings series,
    Taiwan value-chain revenue and 12-month SOX momentum (EM / Kalman). It correlates
    0.95 with WSTS 3-month-average YoY and its Bry&ndash;Boschan turning points line up
    with the record &mdash; 2001, the GFC, the 2019 memory glut, the 2021 shortage boom.</p>
    {img('cycle_factor.png', 'Cycle factor with phase bands and turning points')}
    <div class="tw"><table><thead><tr><th>Phase</th><th>Start</th><th>End</th><th>Months</th></tr></thead>
    <tbody>{chron_rows}</tbody></table></div>
  </section>

  <section>
    <span class="eyebrow">02 &nbsp;/&nbsp; The nowcast</span>
    <h2>Beating an honest benchmark, at the turns</h2>
    <p class="note">Expanding walk-forward with purge and embargo, 240 out-of-sample
    months. The benchmark is an autoregression in point-in-time terms. Adding the
    cycle factor flips LightGBM from losing to the benchmark to beating it at every
    horizon &ge; 3 months &mdash; and by 11&ndash;15% on the months the cycle is actually moving.</p>
    {img('nowcast_oos_h3.png', 'Out-of-sample nowcast vs realised, 3-month horizon')}
    {tbl('scoreboard_h3.csv')}
  </section>

  <section>
    <span class="eyebrow">03 &nbsp;/&nbsp; The strategy</span>
    <h2>A risk overlay, not alpha</h2>
    <p class="note">A cycle-timing weight on SOXX, monthly, 10&nbsp;bps cost. Over
    2005&ndash;2025 it cuts volatility and drawdown but not below buy-and-hold's Sharpe &mdash;
    the ~0.1 gap holds across assets and sub-periods. Semiconductor equities price the
    fundamental cycle before it reaches billings. The deflated Sharpe ratio
    ({rob['deflated_sharpe_ratio']:.2f}) and probability of backtest overfitting
    ({rob['pbo']:.2f}, CSCV) say the positive Sharpe is real but modest.</p>
    {img('strategy_dashboard.png', 'Strategy equity curve, weight over phases, return by phase')}
    <div class="grid2">
      {tbl('strategy_stats.csv', drop=['sortino','calmar','hit_rate'])}
      {tbl('strategy_robustness.csv')}
    </div>
  </section>

  <section>
    <span class="eyebrow">04 &nbsp;/&nbsp; Real-time checks</span>
    <h2>Forward signal &gt; coincident signal</h2>
    <p class="note">Timing off the walk-forward nowcast (a genuine forward forecast)
    beats timing off the coincident factor &mdash; Sharpe 0.88 vs 0.83 on 2009&ndash;2025 &mdash;
    though neither beats buy-and-hold on a window with one real drawdown. Separately,
    the pseudo-real-time factor correlates 0.94 with a fully recursive one, so the
    parameter-fixing shortcut holds.</p>
    {img('realtime_timing_compare.png', 'Nowcast timing vs factor timing vs buy and hold')}
    {tbl('realtime_timing_compare.csv', drop=['months','skew','kurtosis'])}
    {img('realtime_factor_compare.png', 'Pseudo-real-time vs fully recursive factor')}
  </section>

  <section>
    <span class="eyebrow">Brief</span>
    <h2>Where the cycle sits now</h2>
    <div class="brief">{brief}</div>
  </section>

  <footer>
    Built by the <code>semicycle</code> pipeline (Python, DuckDB, statsmodels, LightGBM).
    All data free and public: WSTS, FinMind, FRED, yfinance. Educational and research
    use only &mdash; not investment advice.
  </footer>
</div>
"""

out = load_config().reports_dir / "dashboard_artifact.html"
out.write_text(HTML, encoding="utf-8")
print(out, out.stat().st_size // 1024, "KB")
