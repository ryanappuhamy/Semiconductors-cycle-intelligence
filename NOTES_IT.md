# Diario di bordo — Semiconductor Cycle Intelligence

Note di lavoro e conclusioni, in italiano. Il README (inglese) è la documentazione
"da vetrina"; qui c'è il ragionamento.

---

## Da dove si parte

Progetto precedente (`Semiconductors-cycle-intelligence` su GitHub): idea giusta,
fondamenta fragili. Costruiva un indice di ciclo da 3 z-score (ricavi equipment,
inventory/ricavi, momentum SOXX–QQQ), ma:

- **~5 osservazioni trimestrali utili** — yfinance espone solo pochi trimestri di
  bilanci, e il resto era riempito con spread annuale→trimestrale e
  back-extrapolation. Dati in parte inventati che alimentavano il segnale.
- **nessun modello** — media pesata di z-score con pesi decisi a mano.
- **nessuna validazione** — medie in-sample dei rendimenti forward per fase, con
  celle da n=1.

Questa versione tiene l'intuizione economica (i 3 pilastri, le 4 fasi
Early/Mid/Late/Downturn) e rifà tutto il resto.

---

## Modulo 1 — dati, feature panel, nowcast

### Dati (tutti gratuiti)

| Fonte | Cosa | Storia | Ruolo |
|---|---|---|---|
| WSTS Blue Book | vendite mondiali di semiconduttori, mensili, per regione | 1986– | **target** del nowcast |
| FinMind (ricavi mensili Taiwan) | TSMC, UMC, MediaTek, Nanya, Novatek, GUC (+ ASE dal 2018) — obbligo di deposito entro 10 giorni | 2005– | indicatore fondamentale più tempestivo, ~3 settimane prima di WSTS |
| FRED (CSV pubblico) | IP semiconduttori, nuovi ordini, inventory/sales, PPI, disoccupazione, Nasdaq | 1948– | ampiezza macro |
| yfinance | ^SOX, SOXX, SMH, QQQ, ^GSPC + 15 titoli | 2004– | feature di momentum di mercato + universo strategia |

### La regola anti-look-ahead

Ogni serie ha un `release_lag_days` (WSTS ~35, Taiwan ~11, FRED per-serie).
`features.transforms.as_of_panel` costruisce il panel in modo che la riga del
mese `T` contenga **solo valori con data di pubblicazione ≤ T** — quello che un
analista aveva davvero sullo schermo a fine mese `T`. Il *target*, invece, è il
valore finale rivisto, datato al suo mese di riferimento: vogliamo prevedere cosa
è successo davvero, non la prima stima.

Tre test in `tests/test_transforms.py` bloccano le regressioni su questo punto
(troncare la serie non deve cambiare gli z-score passati; il panel non deve mai
"vedere" un dato pubblicato dopo la data as-of).

### Nota sul Blue Book di giugno 2026

Le righe 2026 di quel file implicano un fatturato mondiale H1-2026 pari a ~2 volte
la dimensione nota del mercato (salti MoM del +20–26%, mai visti in 40 anni di
storia). Quasi certamente un artefatto di data-entry / metodologia di quella
release. Il loader taglia gli "actual" a `2025-12-31` e stampa un warning di
plausibilità. Da rivedere quando esce un Blue Book più recente.

### Validazione

Walk-forward espanding con **purge + embargo** (López de Prado): per ogni mese
out-of-sample `t`, si ri-addestra solo su righe abbastanza vecchie che la loro
finestra-target non tocchi quella di `t`:

```
train:  d  tale che  d + horizon + purge + embargo  ≤  t   (mesi)
```

Benchmark = autoregressione espressa solo in termini disponibili a `t` (il target
WSTS "vecchio di 2 mesi" e i suoi ritardi). I modelli con feature devono batterlo
**out-of-sample**, non in-sample.

Finestra di scoring: **da gennaio 2006**, cioè quando gli indicatori tempestivi
(ricavi Taiwan dal 2005, ETF semiconduttori, IP FRED) hanno qualche anno di
storia. Il benchmark AR è valutato sulla stessa identica finestra.

### Risultati

_(Compilati automaticamente dall'ultimo run di `make nowcast`;
`reports/nowcast_oos_h*.png` e `reports/feature_importance_h*.csv`.)_

<!-- RESULTS:START -->
**h = 0 months** — 240 OOS months (2006-01 … 2025-12)

| model        |    MAE |   RMSE |   skill_vs_AR |   dir_acc |   turn_acc |   corr |   MAE_turns |   turn_acc_turns |
|:-------------|-------:|-------:|--------------:|----------:|-----------:|-------:|------------:|-----------------:|
| ar_benchmark | 0.032  | 0.0444 |        0      |    0.9375 |     0.5272 | 0.9559 |      0.0475 |           0.5823 |
| elasticnet   | 0.0379 | 0.0528 |       -0.1833 |    0.9083 |     0.5565 | 0.9358 |      0.05   |           0.6329 |
| lightgbm     | 0.0385 | 0.0517 |       -0.2025 |    0.925  |     0.5523 | 0.9385 |      0.0531 |           0.557  |

**h = 3 months** — 237 OOS months (2006-01 … 2025-09)

| model        |    MAE |   RMSE |   skill_vs_AR |   dir_acc |   turn_acc |   corr |   MAE_turns |   turn_acc_turns |
|:-------------|-------:|-------:|--------------:|----------:|-----------:|-------:|------------:|-----------------:|
| ar_benchmark | 0.0748 | 0.107  |        0      |    0.8186 |     0.4237 | 0.729  |      0.1065 |           0.4103 |
| lightgbm     | 0.0818 | 0.111  |       -0.0928 |    0.7975 |     0.4703 | 0.6901 |      0.1027 |           0.4615 |
| elasticnet   | 0.0867 | 0.1411 |       -0.1588 |    0.8101 |     0.4746 | 0.6209 |      0.1242 |           0.4359 |

**h = 6 months** — 234 OOS months (2006-01 … 2025-06)

| model        |    MAE |   RMSE |   skill_vs_AR |   dir_acc |   turn_acc |   corr |   MAE_turns |   turn_acc_turns |
|:-------------|-------:|-------:|--------------:|----------:|-----------:|-------:|------------:|-----------------:|
| ar_benchmark | 0.1011 | 0.1392 |        0      |    0.7692 |     0.5408 | 0.4947 |      0.14   |           0.4805 |
| lightgbm     | 0.1017 | 0.1425 |       -0.0058 |    0.7051 |     0.4979 | 0.4521 |      0.1322 |           0.4286 |
| elasticnet   | 0.1236 | 0.1994 |       -0.222  |    0.7094 |     0.4893 | 0.0653 |      0.18   |           0.4545 |

<!-- RESULTS:END -->

### Lettura dei risultati

Il risultato onesto — e il motivo per cui è credibile.

- **Sull'MAE pieno, l'AR benchmark vince a tutti gli orizzonti.** Il target è la
  YoY di una media mobile a 3 mesi: fortemente autocorrelato (corr col proprio
  passato ~0.95 a h=0). Un'autoregressione point-in-time è una baseline durissima,
  e deve esserlo. I modelli con feature le arrivano vicino (h=0: MAE 0.038 vs
  0.032; h=6: LightGBM 0.1017 vs 0.1011, sostanzialmente pari) ma non la battono
  in media. Chi presenta un modello che "straccia" un AR su una serie così liscia
  di solito ha un leak.
- **Dove le feature pagano è alle inflessioni.** Sul terzo di mesi in cui il ciclo
  si muove davvero (`|Δ| ` sopra il 67° percentile):
  - **h = 3** — LightGBM `MAE_turns` 0.103 contro 0.107 dell'AR (**−3.5%**), e
    prende la direzione della variazione più spesso (`turn_acc` 0.47 vs 0.42,
    `turn_acc_turns` 0.46 vs 0.41).
  - **h = 6** — LightGBM `MAE_turns` 0.132 contro 0.140 (**−5.6%**).
  Nei mesi "piatti" l'AR vince semplicemente persistendo l'ultimo valore, e questo
  gli tiene basso l'MAE aggregato. Ma una strategia di ciclo opera proprio intorno
  alle svolte, ed è lì che il modello con feature aggiunge qualcosa.
- **Feature importance.** Domina la struttura regionale WSTS (Asia-Pacific YoY e
  z-score), poi Nasdaq YoY e inventory/sales FRED. I ricavi mensili di Taiwan
  **non** aggiungono informazione ortogonale: la voce "Asia Pacific" di WSTS
  cattura già lo stesso segnale. Il valore del feed Taiwan è di **tempestività**
  (disponibile ~3 settimane prima di WSTS), non di contenuto — cosa che conta per
  un nowcast operativo, non per questo backtest a valori finali.
- **ElasticNet** ha un RMSE alto per via di un overshoot durante il crollo 2008–09
  (poca storia di training, estrapolazione lineare in un crollo non lineare):
  visibile nella figura h=3.

**Implicazione per i moduli successivi.** (1) Il target 3MMA-YoY è troppo liscio:
il fattore latente del Modulo 2 (DFM) dovrebbe dare un segnale di ciclo più pulito
e meno auto-deterministico. (2) Una strategia (Modulo 3) non deve battere l'AR
ovunque — le basta avere ragione alle svolte, ed è quello che i numeri
turn-conditional dicono essere possibile.

### Cosa manca ancora (moduli successivi)

- **Modulo 2:** fattore latente del ciclo con un dynamic factor model a frequenza
  mista (`statsmodels DynamicFactorMQ`) + dating Bry–Boschan → cronologia
  Early/Mid/Late/Downturn data-driven; il fattore rientra come feature del nowcast.
- **Modulo 3:** strategia equity guidata dal ciclo (timing su ^SOX/SMH + tilt
  cross-section), backtest mensile con costi, e statistiche oneste: Sharpe,
  **deflated Sharpe ratio**, **probabilità di overfitting del backtest** (CSCV),
  turnover, performance per regime. Migrazione di `dashboard.py` e `ai_brief.py`.
- **Modulo 4:** vintage ALFRED reali, dashboard HTML, scrittura finale, push su GitHub.
