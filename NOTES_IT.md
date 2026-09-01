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
| elasticnet   | 0.0314 | 0.042  |        0.0188 |    0.9375 |     0.5774 | 0.96   |      0.0451 |           0.6709 |
| ar_benchmark | 0.032  | 0.0444 |        0      |    0.9375 |     0.5272 | 0.9559 |      0.0475 |           0.5823 |
| lightgbm     | 0.039  | 0.0514 |       -0.2182 |    0.9042 |     0.5356 | 0.9391 |      0.0525 |           0.5823 |

**h = 3 months** — 237 OOS months (2006-01 … 2025-09)

| model        |    MAE |   RMSE |   skill_vs_AR |   dir_acc |   turn_acc |   corr |   MAE_turns |   turn_acc_turns |
|:-------------|-------:|-------:|--------------:|----------:|-----------:|-------:|------------:|-----------------:|
| lightgbm     | 0.0722 | 0.0989 |        0.035  |    0.8523 |     0.4915 | 0.7575 |      0.0908 |           0.5128 |
| ar_benchmark | 0.0748 | 0.107  |        0      |    0.8186 |     0.4237 | 0.729  |      0.1065 |           0.4103 |
| elasticnet   | 0.0781 | 0.1255 |       -0.0438 |    0.8186 |     0.4958 | 0.7032 |      0.1096 |           0.4872 |

**h = 6 months** — 234 OOS months (2006-01 … 2025-06)

| model        |    MAE |   RMSE |   skill_vs_AR |   dir_acc |   turn_acc |   corr |   MAE_turns |   turn_acc_turns |
|:-------------|-------:|-------:|--------------:|----------:|-----------:|-------:|------------:|-----------------:|
| lightgbm     | 0.0944 | 0.1353 |        0.067  |    0.7393 |     0.5494 | 0.4889 |      0.1251 |           0.4805 |
| ar_benchmark | 0.1011 | 0.1392 |        0      |    0.7692 |     0.5408 | 0.4947 |      0.14   |           0.4805 |
| elasticnet   | 0.1128 | 0.179  |       -0.1153 |    0.7265 |     0.5107 | 0.2352 |      0.168  |           0.4675 |

<!-- RESULTS:END -->

### Lettura dei risultati

_(La tabella qui sopra è lo stato **dopo** il Modulo 2 — cioè con il fattore del
ciclo tra le feature.)_

- **Modulo 1, solo indicatori: l'AR benchmark vinceva a ogni orizzonte.** Il
  target è la YoY di una media mobile a 3 mesi: autocorrelato ~0.95 col proprio
  passato. Un'autoregressione point-in-time è una baseline durissima, e deve
  esserlo. I modelli con feature le arrivavano vicino ma non la battevano in
  media; l'unico vantaggio era alle inflessioni (LightGBM −3.5% di `MAE_turns` a
  h=3). Chi "straccia" un AR su una serie così liscia di solito ha un leak.
- **Modulo 2, aggiungendo il fattore del ciclo: i modelli con feature passano
  davanti al benchmark.** Vedi sopra: a h=3 lo skill di LightGBM vs AR va da
  −9.3% a **+3.5%**, a h=6 da −0.6% a **+6.7%**; l'errore nei mesi di svolta
  scende del 11–15% rispetto all'AR. E `cycle_factor__chg6` (la variazione a 6
  mesi del fattore) è **la feature #1** di LightGBM a h=6.
- **Perché funziona.** Il fattore latente è una stima del *segnale comune* dietro
  i 6 indicatori — ripulita dal rumore idiosincratico di ciascuno. Preso da solo,
  nessuno dei 6 indicatori porta quella informazione: è proprio il punto di
  stimarlo. Il target 3MMA-YoY resta liscio e sull'MAE aggregato l'AR è ancora
  competitivo nei mesi piatti; ma alle svolte — dove una strategia di ciclo
  opera — ora c'è un margine reale.
- **ElasticNet** ha ancora un RMSE alto per un overshoot durante il crollo
  2008–09 (poca storia, estrapolazione lineare in un crollo non lineare).

---

## Modulo 2 — il fattore latente del ciclo

### Dal peso a mano alla stima

Il progetto vecchio faceva `indice = 0.40·z₁ + 0.30·z₂ + 0.30·z₃`, pesi decisi a
occhio. Qui la componente comune è **stimata** con un dynamic factor model:

```
x_it = λ_i · f_t + e_it            f_t = a₁·f_{t-1} + a₂·f_{t-2} + u_t
```

`f_t` è l'indice del ciclo; `λ_i` (i carichi) e le dinamiche AR sono stimate per
massima verosimiglianza (EM, `statsmodels DynamicFactorMQ`). Il modello gestisce
da solo il **bordo frastagliato**: gli indicatori finiscono in mesi diversi
perché escono con ritardi di pubblicazione diversi.

### Input

I 4 billings regionali WSTS (YoY) + ricavi Taiwan (YoY) + momentum a 12 mesi del
SOX. Tutti sguardi coincidenti sulla domanda **globale** di semiconduttori.

Nota di percorso: avevo provato IP semiconduttori USA e nuovi ordini FRED —
**scartati**. La produzione di fab USA è un pezzo piccolo e vincolato dalla
capacità: è rimasta piatta durante il boom 2021, tirando il fattore fuori dal
ciclo (corr col target crollava da 0.95 a 0.66).

### Il fattore

Correlazione **0.95** con la misura "nota" (WSTS 3MMA YoY). Carichi bilanciati
(0.36–0.49). Combacia con la storia: picco 2000, crollo 2001, GFC, glut memoria
2019, boom 2021, minimo 2023.

### Dating — Bry–Boschan

`cycle/dating.py` applica l'algoritmo classico per datare picchi/minimi
(estremi locali alternati, fasi ≥ 5 mesi, cicli ≥ 18 mesi, censura ai bordi) e
mappa svolte + linea dello zero sulle 4 fasi:

```
minimo→picco (espansione):  Early sotto 0,  Mid sopra 0
picco→minimo (contrazione):  Late sopra 0,  Downturn sotto 0
```

**22 punti di svolta, 35 fasi dal 1987 al 2026**, ~14 mesi di fase media → ciclo
completo ~4 anni. Fase attuale: **Mid Cycle**, fattore **+1.7**.
Figura: `reports/cycle_factor.png`. Cronologia: `reports/cycle_chronology.csv`.

### Il fattore nel nowcast — versione pseudo-real-time

Per usarlo come feature senza look-ahead: input costruiti point-in-time
(`as_of_panel`), e si prende lo stato **filtrato** di Kalman (stima di `f_t` con
i soli dati fino a `t`). I parametri restano full-sample — è l'approssimazione
"pseudo real-time" standard nella letteratura di nowcasting. Entra nel pannello
come `cycle_factor`, `cycle_factor__chg3`, `cycle_factor__chg6`.

---

## Modulo 3 — la strategia (e il risultato onesto)

### Il segnale e il backtest

Dal segnale point-in-time del ciclo (`cycle_factor` + la sua variazione a 6 mesi,
z-score a finestra espansiva) a un peso target su SOXX:

```
peso_T = clip( base + gain · segnale_T ,  0 ,  max )      base=1.0  gain=0.4  max=1.25
```

Ribilanciamento mensile, **costo 10 bps** sul turnover. Convenzione temporale
senza look-ahead: il peso deciso a fine mese `t` frutta il rendimento del mese
`t+1`; il costo del trade grava su `t+1`.

Nota dati: i prezzi yfinance dei semiconduttori mostrano movimenti mensili
impossibili da aprile 2026 (ETF che si muovono 2x il loro titolo maggiore) — la
stessa era dei dati WSTS 2026 corrotti. Backtest tagliato a dicembre 2025.

### Il risultato (2005–2025, vs buy & hold SOXX)

| | strategia | buy & hold |
|---|---|---|
| rendimento ann. | +10.9% | +15.5% |
| volatilità ann. | 21.6% | 25.0% |
| **Sharpe** | **0.59** | **0.70** |
| max drawdown | −54% | −60% |

**La strategia NON batte il buy & hold in termini risk-adjusted.** Taglia
volatilità e drawdown, ma il rendimento cala di più: lo Sharpe resta ~0.1 sotto.
E questo gap è stabile: vale con SMH, con un basket equal-weight, e in ogni
sotto-periodo. Per regime: +23% ann. in Early, +20% in Late, **−8% in Downturn**
— il de-risking funziona nella direzione giusta, ma non abbastanza.

**Perché.** Le azioni semi sono *forward-looking*: prezzano il ciclo fondamentale
prima che si veda nei billings. Un segnale coincidente sul ciclo arriva tardi per
il timing azionario (si de-risk nel minimo e si perde il rimbalzo — visibile
nella equity curve, 2009–2013). Il valore del segnale è come **overlay di
rischio**, non come fonte di alfa.

### Le due statistiche anti-overfitting

- **Deflated Sharpe ratio = 0.99** — probabilità che lo Sharpe *vero* sia > 0
  dopo aver considerato le 16 config della griglia. Alto perché le 16 config
  danno risposte simili (poca "ricerca" da scontare) e lo Sharpe positivo è
  robusto — *non* perché batta il benchmark.
- **PBO (CSCV) = 0.32** — ~1/3 di probabilità che la config scelta sia
  overfittata rispetto alle altre. Moderato. La lezione: leggere la strategia
  come regola di de-risking, non andare a caccia della config che "vince".

### Migrazione dal v1

`dashboard.py` → `report.plots.strategy_dashboard` (4 pannelli: equity, peso sulle
fasi, rendimento per fase, scheda statistiche). `ai_brief.py` → `report.brief`
(prompt strutturato → Claude, con fallback a template locale se manca
`ANTHROPIC_API_KEY`).

---

## Cosa manca ancora

- **Modulo 4:** provare a fare il timing sul **nowcast forward** (billings +3/+6
  mesi) invece che sul fattore coincidente; fattore real-time completamente
  ricorsivo (parametri ri-stimati ogni mese); vintage ALFRED; dashboard HTML;
  scrittura finale.
