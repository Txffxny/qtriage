# qtriage

**Statistically certified alert triage for security operations.**

Anomaly detectors produce scores. Scores get thresholded. The threshold is
chosen by whoever tuned it last, and nobody can say what fraction of the
resulting alerts are false. This project asks whether the multiple-testing
machinery used in genomics — where the same problem appears as "which of 18,000
genes are differentially expressed" — can put a stated false-discovery bound on
a SOC alert queue.

The short answer: **yes, but the guarantee bounds a quantity that is not what an
analyst experiences, and on real traffic the two differ by a factor of twelve.**

---

## Method

Conformal p-values calibrated on attack-free traffic, corrected with
Benjamini–Hochberg within time batches.

1. Fit an IsolationForest on one half of a benign baseline day.
2. Score the other half to obtain a null distribution of anomaly scores.
3. For a test unit, the conformal p-value is the fraction of calibration scores
   at least as extreme: `p = (1 + #{cal ≥ s}) / (n_cal + 1)`.
4. Apply BH within each time batch to control the false discovery rate.

Conformal p-values are marginally valid under exchangeability and are PRDS, so
BH controls FDR over the batch (Bates, Candès, Lei, Romano & Sesia, *Testing for
outliers with conformal p-values*, Annals of Statistics 51(1), 2023).

Crucially, the fit and calibration halves are disjoint. Scoring a detector's own
training points yields in-sample scores that are not exchangeable with test
scores, which would silently invalidate every p-value downstream.

## Data

The [corrected CICIDS2017 release](https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html)
(Engelen, Rimmer & Joosen, IEEE SPW 2021), which regenerates flows with a fixed
CICFlowMeter and relabels over 20% of the original dataset. The original
labelling is not usable here: the headline metric is realised FDP measured
against ground truth, so mislabelled flows corrupt the measurement directly.

Monday (371,749 flows, 100% benign) is the calibration baseline. Tuesday–Friday
are test days. Flows labelled `X - Attempted` — payload-reliant attacks where no
payload was delivered — are excluded by default, so that FDP measures what it
claims to. `ATTEMPTED_POLICY` in `src/config.py` switches this for sensitivity
analysis.

---

## Results

### 1. Flow-level detection fails, and AUC does not reveal it

| metric | value |
|---|---|
| AUC (Tuesday, brute-force traffic) | **0.799** |
| attacks detected under FDR control, any q from 0.01–0.30 | **0** |
| most anomalous attack flow, percentile vs benign | 97.9th |
| benign flows scoring below p = 0.01 | 3,361 |

BH at m ≈ 38,000 hypotheses requires p-values near 1.3 × 10⁻⁶ — roughly the
99.9999th percentile. The best attack flow reached the 97.9th. Four orders of
magnitude short.

**AUC is a rank-average statistic; conformal FDR control is a tail statistic.**
A detector can be respectable on the first and useless on the second. Most
reported NIDS performance lives in that gap.

The mechanism is specific to the attack class. Brute-force authentication
produces flows that are short, well-formed and utterly ordinary. What is
anomalous is the *pattern* — thousands of near-identical connections to one port
— not any individual flow. Outlier detection is the wrong frame for it.

### 2. Aggregation recovers the signal

Scoring `(source host, 5-minute window)` units instead of individual flows,
with features capturing repetition rather than rarity: destination-port entropy,
coefficient of variation of flow duration and byte counts, inter-arrival
regularity, RST rates.

| metric | flow-level | host-window |
|---|---|---|
| AUC | 0.799 | **0.987** |

Same detector, same statistics, same data. Only the unit of analysis changed.

This also repairs a dependency problem: thousands of flows from one attack were
never independent hypotheses. One host-window genuinely is one test.

### 3. A feasibility rule falls out of the failure

With `H` hosts and `W` windows in the calibration period, BH can reject a single
unit only if the conformal p-value floor sits below the k = 1 threshold:

```
n_cal = H·W/2 ,  m = H  →  2/(H·W) ≤ q/H  →  W ≥ 2/q
```

**`H` cancels.** Network size is irrelevant; window count is everything. At
q = 0.05 the calibration period needs at least 40 windows. Hourly batching over
one day gives 10 — blocked, and blocked identically on a 5,000-host network.
Five-minute windows give 96, which works.

### 4. The guarantee holds — and bounds the wrong thing

A null test (benign-only data, so every rejection is false by construction)
confirms the p-values are valid: KS vs uniform p = 0.593 on a random within-day
split, and BH fires in 2.0% of batches at q = 0.05 across days. Exchangeability
is not the problem.

Yet three different false discovery proportions can be computed from the *same*
rejections, averaged over Tuesday–Friday:

| q | all batches | firing batches only | pooled over the day |
|---|---|---|---|
| 0.05 | 0.024 | **0.300** | 0.310 |
| 0.10 | 0.052 | 0.319 | 0.385 |
| 0.20 | 0.119 | 0.496 | 0.573 |
| 0.30 | 0.201 | 0.607 | 0.671 |

BH bounds the first column, and does so correctly. But roughly 90% of batches
never fire, and a batch making no discoveries has a false discovery proportion
of zero *by definition* — true and informationally empty. Firing batches reject
1.28 units on average, so their FDP is close to binary.

**Conditional on the system raising an alert, 30% of alerts are false at a
nominal q of 0.05.** All three numbers are correct. Only one answers the
question an analyst is asking.

### 5. Stream-level control is out of reach on one day of baseline

To make the guarantee apply to the queue rather than the batch, the hypothesis
family must be the whole stream: `m = H·W`, requiring `n_cal ≥ m/q`. With `D`
calibration days, `n_cal = H·W·D/2`, so `D ≥ 2/q` — **41 benign days at
q = 0.05**, measured directly from this data. CICIDS2017 contains one.

This is not a tuning problem. No configuration closes it.

---

## What this implies for building the thing

Two channels, one statistical engine, different guarantees:

- **Channel A — rarity (flow-level).** Null: *this flow is drawn from the benign
  baseline*. Rejecting it means "statistically unusual", which is true and
  useful. Routes to a low-priority review queue. Never to an automated block.
- **Channel B — behaviour (host-window).** Null: *this host behaved normally
  this window*. Targets attack patterns. Primary alert channel.

And a display rule: **ranking and certification are separate.** Every unit has a
p-value, so the ranked queue always exists. BH does not produce the ranking; it
draws a line on it with a bound attached. A dashboard should show both — the
queue, and the line below which no guarantee is available — rather than implying
a certificate over the whole list.

---

## Limitations

- **One baseline day.** Every result calibrates on the same Monday, so the four
  test days are not fully independent replicates.
- **Small networks.** 16 calibration hosts, 18 test hosts. The effective
  independent sample size is closer to the host count than the window count.
- **Attacker hosts are absent from the baseline.** In CICIDS2017 attacker IPs
  appear only on attack days, so malicious host-windows differ from benign ones
  in provenance as well as behaviour. The AUC of 0.987 is measured on that
  comparison and should be read with the confound in mind.
- **28 malicious host-windows on Tuesday.** Recall estimates on the sparse days
  carry wide error bars and should not be read to three decimals.
- **Mild across-day drift.** KS D = 0.059 Monday→Tuesday on benign traffic:
  detectable, conservative in direction, not enough to break FDR control here.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/validate_stats.py      # synthetic check: BH controls FDR
python scripts/build_cache.py         # raw CSV -> parquet, medians fitted on Monday
python scripts/inspect_hosts.py       # host-window cardinality and feasibility
python scripts/null_test.py           # are the p-values valid?
python scripts/run_three_metrics.py   # the headline result
streamlit run app.py                  # dashboard
```

Place the corrected CICIDS2017 CSVs in `data/raw/`. Results and figures are
written to `figures/<experiment>/`, each with a `run_meta.json` recording the
settings that produced it.

## Layout

```
src/        importable library: stats, data loading, aggregation, detector
scripts/    entry points: one experiment each, writes to figures/
figures/    results, one directory per experiment, never overwritten
app.py      Streamlit dashboard
```

`src/` holds functions you import; `scripts/` holds files you run. That split is
why the conformal code validated on synthetic data in `validate_stats.py` is the
identical code running on network flows, unchanged.
