# Knowledge-Graph-Driven Credit Risk & Credit-Increase Scenario Analysis

A proof of concept that asks one question:

> **Can representing a customer's six-month credit behaviour as a temporal knowledge graph, and extracting graph-derived behavioural features, provide useful information for credit-risk modelling and for explainable credit-increase analysis?**

The project goes from raw data to validated patterns, a Neo4j temporal graph, Cypher-derived features, a baseline-vs-graph-enhanced model comparison, and finally a transparent, *hypothetical* credit-increase scenario score with per-customer explanations.

> ⚠️ **The credit-increase score is a demonstrative scenario framework, not a banking underwriting policy.** The dataset contains no credit-increase requests or outcomes. The requested increase is a hypothetical, configurable scenario parameter. `PAY_AMT` columns are **historical payment behaviour**, never income.

---

## Contents
1. [Short answer](#1-short-answer)
2. [Dataset and what inspection revealed](#2-dataset-and-what-inspection-revealed)
3. [Architecture](#3-architecture)
4. [Knowledge graph schema](#4-knowledge-graph-schema)
5. [The nine layers (notebooks)](#5-the-nine-layers-notebooks)
6. [Results](#6-results)
7. [Credit-increase scenario](#7-credit-increase-scenario)
8. [How to run](#8-how-to-run)
9. [Project structure](#9-project-structure)
10. [Validation and tests](#10-validation-and-tests)
11. [Limitations](#11-limitations)
12. [Glossary](#12-glossary)

---

## 1. Short answer

| Question | Answer |
|---|---|
| Do behavioural patterns in six months of history relate to next-month default? | **Yes, for some.** Delinquency, payment coverage, credit limit and typical payment size are strongly related. Utilization is U-shaped. "Rising utilization = deterioration" is **not** supported. |
| Does the graph find meaningful patterns without seeing the label? | **Yes.** For example, late 3+ months in a row → 60% default (2.7× base); late → recovered → late again → 54%; always paid in full → 12%. Train and test agree. |
| Do graph features improve a **Random Forest**? | **No.** ΔROC-AUC = −0.0008, 95% CI [−0.0019, +0.0004]. The forest already learns order patterns from the raw monthly columns. |
| Do graph features improve an **interpretable logistic regression**? | **Yes, reliably.** ΔROC-AUC = +0.0065, 95% CI [+0.0027, +0.0101]; ΔPR-AUC = +0.018. B beats A in 24 of 25 CV folds. |
| Could a graph neural network do better? | An exploratory GNN that learns **only from the graph** reaches 0.783 AUC vs 0.788 for the engineered baseline: close, but not better, and it adds nothing in an ensemble. All approaches converge at ~0.78–0.79, the information limit of this dataset. |
| Does the relative credit-increase framing work? | **Yes.** With a fixed absolute request, only ~16% of low-limit customers are "supportable" vs ~65% of high-limit ones. With a request scaled to each customer's own typical payment it is ~57% vs ~68%, so the gap reflects behaviour, not size. |
| Is the score fair across demographic groups? | **Close to it with own-scale requests, not with fixed ones.** Demographics are never inputs, but outcomes were audited (notebook 09). With 3 × typical payment, every group's "supportable" rate is ≥ 0.83 of the top group's. With a fixed 30,000, 21–24-year-olds get it at 0.48 of the rate of 25–34-year-olds, because their limits are far smaller. Nothing was "fixed" silently; see the report. |

---

## 2. Dataset and what inspection revealed

**Source:** [Default of Credit Card Clients (UCI / Kaggle)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset), in `dataset/archive/UCI_Credit_Card.csv`.
30,000 customers × 25 columns (Taiwan, April–September 2005). No missing values; unique IDs.

**Target (real, never replaced):** `default.payment.next.month`, renamed `default_next_month`. 22.12% positive (6,636 / 30,000).

| Raw columns | Meaning | Used as |
|---|---|---|
| `LIMIT_BAL` | credit limit | `credit_limit` |
| `PAY_0, PAY_2 … PAY_6` | repayment status Sep → Apr | `status_m6 … status_m1` + named repayment state |
| `BILL_AMT1 … 6` | statement balance Sep → Apr (can be negative) | `bill_m6 … bill_m1` |
| `PAY_AMT1 … 6` | amount paid in Sep → Apr | `pay_m6 … pay_m1` (payment behaviour, **not income**) |
| `SEX, EDUCATION, MARRIAGE, AGE` | demographics | descriptive only; **excluded** from models and the score |

Months are renamed in time order: **m1 = April … m6 = September 2005**.

### Facts found by inspecting the data (they shape the design)

| Finding | Evidence | Consequence |
|---|---|---|
| **A payment in month t pays the bill of month t−1** | `PAY_AMTk` equals `BILL_AMT(k+1)` in 4,066–4,616 rows per month vs 582–671 for the same month | Payment coverage = `pay_t / bill_(t-1)`, 5 pairs per customer; stored on the graph's `NEXT_MONTH` edge |
| Status code `1` almost vanishes after September | Code 1: 3,688 in PAY_0; 28 / 4 / 2 / 0 / 0 in PAY_2…6 | Codes are treated as **ordered labels**, not exact month counters; `late = status ≥ 1` |
| Delays are not a clean running counter | 28% of late months jump by more than +1 from the prior month | Named states instead of "months overdue" arithmetic |
| −2 / −1 / 0 are not ordered by risk | September default rate 13.2% / 16.8% / 12.8% | States `NO_USE`, `PAID_FULL`, `REVOLVING` are kept separate |
| Utilization vs default is **U-shaped** | 36% default at 0 utilization, 16% at 10–30%, 35% above 100% | Inactive accounts are not rewarded; "insufficient evidence" band |
| Odd values are real | 3,932 negative bill cells; 8,274 cells with utilization > 100%; undocumented education/marriage codes | Kept and handled explicitly, never deleted |

---

## 3. Architecture

```
dataset/archive/UCI_Credit_Card.csv (raw, read-only)
        │
 01 prepare ─────► data/processed/customers.parquet · monthly.parquet (180k rows) · split.csv (80/20, seed 42)
        │
 02 EDA ─────────► outputs/figures/eda_*.png
 03 validate ────► outputs/tables/pattern_validation.csv ─► data/features/feature_spec.json (frozen)
        │                                              └─► data/features/baseline.parquet
        │
 04 graph ───────► Neo4j: Customer · CreditAccount · MonthlyCreditRecord · RepaymentState
        │           (idempotent MERGE load, constraints, NO target label in the graph)
 05 Cypher ──────► pattern evidence (Q1–Q11) ─► validated in Python ─► data/features/graph.parquet
        │           (parity-tested against pandas)
        ▼
 06 Experiment A (baseline RF) ─┐ same split, same 25 folds, same hyper-parameters,
 07 Experiment B (+ graph)     ─┴► same threshold procedure ─► bootstrap CI  (+ B-LR, calibrated PD)
        ▼
 08 Experiment C: hypothetical increase (absolute or relative to own scale)
        + calibrated PD + validated behaviour ─► suitability score + explanation (config/scoring.yaml)
        ▼
 main.py explain --customer ID [--increase N | --times-typical-payment k | --share-of-limit p]
```

---

## 4. Knowledge graph schema

```
(:Customer {customer_id, age, sex_code, education_code, marriage_code, split})
   -[:HAS_ACCOUNT]->
(:CreditAccount {account_id, credit_limit})
   -[:HAS_MONTHLY_RECORD]->
(:MonthlyCreditRecord {record_id "<id>-<m>", month_index 1..6, month, bill_amount, payment_amount,
                       repayment_status, state, utilization, is_late, credit_balance})
   -[:NEXT_MONTH {coverage, util_delta, status_change, transition}]->
(:MonthlyCreditRecord)
(:MonthlyCreditRecord)-[:IN_STATE]->(:RepaymentState {code, ordinal, description})
```

| Design decision | Reason |
|---|---|
| **Coverage lives on the `NEXT_MONTH` edge** | A payment in month t+1 settles the bill of month t, so it describes the transition, not either month. |
| **5 shared `RepaymentState` nodes** | All 180k monthly records link to them, which yields a population state-transition network (learned from **training customers only**). |
| **No target in the graph** | Cypher can only discover behaviour; default rates are joined in Python afterwards. Checked automatically. |
| **No demographic nodes** | They add hubs with no reasoning value and invite demographic "patterns". Kept as properties for description only. |
| **Idempotent load** | Uniqueness constraints first, then `MERGE` by key in batches of 5,000. Loading twice gives identical counts (tested). |

**Size:** 240,005 nodes (30k customers, 30k accounts, 180k monthly records, 5 states) and 540,000 relationships. Loads in about 20 seconds.

**Try it in Neo4j Browser:**
```cypher
MATCH p=(:Customer {customer_id: 1})-[:HAS_ACCOUNT]->()-[:HAS_MONTHLY_RECORD]->
        (:MonthlyCreditRecord {month_index: 1})-[:NEXT_MONTH*5]->()
RETURN p
```

---

## 5. The nine layers (notebooks)

Each notebook has a short plain-language explanation above every code cell and ends with a *Takeaways* section.

| # | Notebook | What it does | Main outputs |
|---|---|---|---|
| 01 | `01_data_preparation` | Load and check the CSV, prove the payment/bill offset, reshape to wide + monthly (180k) tables, fixed stratified split | `data/processed/*`, `data_quality.csv` |
| 02 | `02_eda` | Distributions, default rate by behaviour bucket with CIs, transitions, trajectories, correlations, interactions (training split only) | `eda_*` figures and tables |
| 03 | `03_pattern_validation_features` | 13 hypotheses judged with trend tests, Wilson CIs and single-feature AUC; redundancy and weak-signal pruning; frozen feature spec | `pattern_validation.csv`, `feature_spec.json`, `docs/FEATURES.md`, `baseline.parquet` |
| 04 | `04_graph_construction` | Constraints, batched idempotent load, integrity checks (counts, time order, no target, sample match, run-twice) | Neo4j graph |
| 05 | `05_graph_patterns_features` | Cypher patterns Q1–Q11 with evidence, validation against the real outcome, transition network, 9 graph features, pandas parity test | `graph_pattern_validation.csv`, `graph.parquet` |
| 06 | `06_baseline_model` | Experiment A: tuned Random Forest, 5×5 repeated CV, OOF threshold, single test evaluation, calibration check, permutation importance | `model_A*`, `cv_folds_A*.csv` |
| 07 | `07_graph_enhanced_model` | Experiment B vs A (same folds), corrected t-test, paired bootstrap, ablations, **B-LR** (logistic regression), calibrated PD for the scenario | `comparison*.json`, `pd_calibrated.parquet` |
| 08 | `08_credit_increase_scenario` | Experiment C: relative scenario features, score, gates and bands; absolute vs relative requests; matched real pair; explanations with graph evidence; validation without ground truth | `scenario_*` tables and figures |
| 09 | `09_final_evaluation` | Health check (data, graph, tests, wording), ablation table with corrected CV and bootstrap intervals, calibration, score-vs-outcome checks (including with the model removed), fairness audit by sex, age, education and marital status, limitations, conclusion | `final_*` and `fairness_*` tables and figures; `outputs/reports/final_report.md` and `phase9_results.json` (report generated from the results, never hand-edited) |

---

## 6. Results

### 6.1 Validated patterns (notebook 03, training split)

| Hypothesis | Verdict |
|---|---|
| More late months → more default | **Supported** (11.5% at 0 late months → 69.7% at 6) |
| Recent lateness matters more than old lateness | **Supported** (one late month: 33% if recent vs 21% if earlier, p ≈ 2e-12) |
| Higher payment coverage → less default | **Supported** (58% at zero coverage → ~15% at 50%+) |
| Higher credit limit / larger typical payment → less default | **Supported** |
| Higher utilization → more default | **Only among active users**; overall U-shaped |
| Rising utilization or rising bills → more default | **Not supported (opposite direction)** |
| Payment trend / payment consistency | Weak |
| A request that is large relative to typical payment → more default | **Supported** (with a fixed request this is a transform of typical payment) |

### 6.2 Graph patterns (notebook 05). Found by Cypher, validated outside the graph

| Pattern | Customers | Default rate, train | Default rate, test | Lift |
|---|---|---|---|---|
| Q2 late ≥ 3 months in a row | 3,297 | 60.4% | 60.4% | 2.7× |
| Q1 late in ≥ 3 of 6 months | 3,744 | 59.9% | 60.3% | 2.7× |
| Q10 relapse: late → recovered → late | 1,073 | 54.4% | 52.5% | 2.5× |
| Q8 ≥ 2 concerning signals | 8,149 | 42.9% | 41.8% | 1.9× |
| Q3 utilization ≥ 90% in ≥ 3 months | 5,062 | 30.5% | 28.5% | 1.4× |
| Q6 paid < 10% of statement in ≥ 3 months | 15,433 | 26.7% | 26.1% | 1.2× |
| Q5 recovered, clean in last 3 months | 1,629 | 23.5% | 18.4% | 1.1× |
| Q4 utilization rose ≥ 3 months running | 1,300 | 19.3% | 22.5% | 0.9× |
| Q7 always paid in full, never late | 3,652 | 11.8% | 12.0% | 0.5× |

**Graph features (9):**
- `g_max_late_streak`, `g_late_onsets`, `g_recoveries`, `g_relapses`, `g_latest_transition`, `g_max_util_rise_run` and `g_weak_cov_after_late` come from path and transition queries.
- `g_markov_next_late_prob` and `g_path_typicality` come from the population transition network (training customers only).

All 9 match a pandas re-implementation exactly. That parity test caught and fixed one Cypher bug during development.

### 6.3 Baseline vs graph-enhanced (notebooks 06–07)

Random Forest (`min_samples_leaf=50`, `max_features=0.3`, balanced class weights), fixed 80/20 split, identical 5×5 CV folds, decision threshold chosen on out-of-fold predictions (best F1), test set used once.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A0: raw monthly columns | 0.7845 | – | – | – | – | – |
| **A: baseline** (raw + engineered) | 0.7880 | 0.7800 | 0.5575 | 0.498 | 0.596 | 0.542 |
| **B: + graph features** | 0.7879 | 0.7792 | 0.5566 | 0.501 | 0.601 | 0.546 |
| A + demographics (ablation) | 0.7887 | – | – | – | – | – |
| Graph features only (+ limit) | 0.7704 | – | – | – | – | – |
| **LR-A:** logistic regression, baseline | 0.7665 | 0.7573 | 0.5021 | 0.483 | 0.558 | 0.518 |
| **LR-B:** logistic regression, + graph | 0.7717 | 0.7638 | 0.5197 | 0.506 | 0.545 | 0.525 |

| Comparison | ΔROC-AUC (test) | 95% CI | CV folds B > A | Verdict |
|---|---|---|---|---|
| RF: B − A | −0.0008 | [−0.0019, +0.0004] | 12 / 25 (p = 0.82) | **No reliable improvement** |
| LR: B − A | +0.0065 | [+0.0027, +0.0101] | 24 / 25 (p = 0.001) | **Reliable improvement** |

Accuracy is not a primary metric: predicting "no default" for everyone already gives 77.9%.

**Why the Random Forest does not improve:** each customer's graph is a 6-node chain built from the same 23 columns the model already sees. A tree ensemble learns streaks and recoveries from the raw monthly columns directly, so the graph features are informative (0.770 AUC alone) but redundant. A linear model *cannot* build those order patterns, which is why the graph helps logistic regression.

**Graph neural network (exploratory, `experiments/gnn_prototype.py`):** message passing over each customer's subgraph, read directly from Neo4j (account node, month nodes, `NEXT_MONTH` edge features, learned `RepaymentState` embeddings). 5-fold CV on training data: **GNN 0.7829** vs RF-A 0.7883, average of the two 0.7884. It is close to expert feature engineering with no hand-built features, but gives no gain. Results in `outputs/tables/gnn_prototype_cv.csv`.

**Probability calibration:** isotonic calibration reduced the Brier score from 0.173 to 0.135. Following a rule fixed in advance ("use the simpler model unless the richer one is reliably better"), the scenario uses calibrated **model A** probabilities, out-of-fold for training customers.

### 6.4 Final evaluation (notebook 09)
Full numbers and tables: [`outputs/reports/final_report.md`](outputs/reports/final_report.md), generated from `outputs/reports/phase9_results.json`.
- **Ablation with intervals:** every feature set has a corrected 95% CV interval, and the two test-set comparisons have paired bootstrap intervals. The RF verdict ("no reliable difference") and the LR verdict ("reliable improvement") hold. The GNN is reliably below the engineered forest (−0.0053, CI [−0.0089, −0.0018]).
- **Calibration (test):** the calibrated probability's Brier score is 0.1346 against 0.1723 for always predicting the base rate, with an expected calibration error of 0.011 and slope 0.95. The raw forest score is slightly *worse* than the base-rate predictor, because balanced class weights inflate it.
- **Score vs outcomes with the model removed completely** (risk factor *and* the PD ≥ 0.5 safety check): 11.4% → 29.8% → 65.8% missed the next payment for Supportable → Review → Not supportable (3 × typical payment), so the behavioural factors carry risk information by themselves.
- **Fairness audit:** see section 1. Within the Supportable label, high-school customers miss more often than graduates (13.1% vs 9.6%), mostly visible to the model. Men are slightly under-predicted (12.7% observed vs 11.8% predicted).

---

## 7. Credit-increase scenario

### Principle: the customer's own historical scale decides how big a request is
A request is judged **relative to the customer's own demonstrated behaviour**, never by its absolute size. Any fixed amount (10k, 30k, …) is only an example. For example, 30,000 is ~2.6 months of typical payment for the 5th-percentile customer and ~92 months for the 95th.

### Scenario features (prefixed `scenario_`, never used by the ML models)
| Feature | Definition | Scored? |
|---|---|---|
| `scenario_payback_months` | request ÷ typical (median) monthly payment | yes |
| `scenario_increase_to_limit` | request ÷ current limit | yes |
| `scenario_projected_util` | latest bill ÷ (limit + request) | no, mechanical under a static-bill assumption |
| `scenario_typical_payment_to_increase` | inverse of payback | display only |

Requests can be absolute or **relative to the customer's own scale**: *k × typical payment* or *p × current limit*.

### Score (`config/scoring.yaml`)
Score = 100 × Σ weight × sub-score, where each sub-score ∈ [0, 1] comes from data-anchored breakpoints.

| Component | Weight | Anchor |
|---|---|---|
| Calibrated default probability | 0.30 | the calibrated PD is already an expected default rate |
| Repayment record (late months, halved if latest month late) | 0.20 | H2, H11 |
| Payment coverage of previous statements | 0.15 | H3 |
| Payback months (request ÷ typical payment) | 0.20 | H10 gradient: ≤1 month 4.6% … >30 months 28.6% default |
| Request ÷ limit | 0.10 | **policy assumption**, not data-derived |
| Current utilization (U-shape aware; inactive = neutral) | 0.05 | H1 |

**Gates:**
- PD ≥ 0.5, or latest month 2+ months late → *Not supportable in this scenario*.
- Inactive account, or no payments → *Insufficient behavioural evidence*.

**Bands:** ≥ 70 *Supportable in this scenario*, 50–70 *Review*, < 50 *Not supportable in this scenario*.

### Key scenario results
| "Supportable" share | Absolute 30,000 for everyone | 3× own typical payment | 25% of own limit |
|---|---|---|---|
| Low-limit third | 15.6% | 57.4% | 46.0% |
| Mid-limit third | 53.6% | 62.7% | 52.2% |
| High-limit third | 64.5% | 68.1% | 55.2% |

- **Matched real pair.** Same limit (160k), same PD (~0.06), same coverage and utilization; typical payments 20k vs 10k. The lower payer scores lower for the same request, and the gap widens with the request: 99.1 vs 97.0 at 30k, 91.4 vs 86.5 at 100k.

### Example explanation
```text
$ python main.py explain --customer 102 --times-typical-payment 3
Relative scenario: 3 x typical monthly payment (1,224) = 3,670

Customer 102  |  SCENARIO: hypothetical credit increase of 3,670 (not an observed request)
Current limit 380,000 -> projected 383,670

Historical evidence (Apr-Sep 2005):
  average utilization 0.06 | latest utilization 0.00 | utilization trend -0.020/month
  median coverage of previous statement 0.04 | weak-coverage months 3/5
  late months 0/6 (last 3 months: 0) | latest status code -2
  typical (median) demonstrated monthly payment 1,224
Graph evidence (Neo4j):
  late in: none
  weak payment against previous statement in: 2005-04, 2005-05, 2005-06
  longest late streak 0 | relapses 0 | population P(late next month) 0.08
Scenario-relative evidence:
  request / current limit 0.01 | payback 3.0 months of typical payment
  projected utilization 0.00 (mechanical, assumes the bill stays the same; not scored)
Model: default probability 0.11 (calibrated, out-of-sample)

Suitability score 83/100  ->  Supportable in this scenario
  supports:       predicted default risk (calibrated) (+14.2); repayment record (late months, latest status) (+10.0);
                  request vs typical demonstrated payment (payback months) (+7.0); request size vs current limit (+5.0);
                  current utilization (U-shape aware) (+2.0)
  weighs against: payment coverage of previous statements (-5.3)

Note: Demonstrative scenario analysis only. The requested increase is hypothetical and the score is not an
underwriting decision; the dataset contains no credit-increase outcomes.
```
Every number is read from a computed column, and the listed contributions add up exactly to the score (tested).

---

## 8. How to run

### Prerequisites
- Python 3.11+ (developed on 3.14)
- Neo4j 5.x (developed on 5.26 Community) running locally, needed for notebooks 04, 05 and 08 and for graph evidence in `main.py`. **Use an empty database:** notebook 04 stops if it finds labels from another project.

### Neo4j in Docker (current setup)
Neo4j 5.26 Community runs in a Docker container that restarts with Docker Desktop. Data lives in the `credit-kg-neo4j-data` volume. To recreate it, read the password from `.env` into `NEO4J_AUTH=neo4j/<password>`, then:
```powershell
docker run -d --name credit-kg-neo4j --restart unless-stopped -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH -e NEO4J_server_memory_heap_max__size=2G -e NEO4J_server_memory_pagecache_size=1G `
  -v credit-kg-neo4j-data:/data neo4j:5.26-community
```
Then run notebook 04 to load the graph (about 1 minute). If the graph tools report "Couldn't connect to localhost:7687", start Docker Desktop.

### Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env        # then edit NEO4J_PASSWORD
```
`.env`:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your password>
NEO4J_DATABASE=
```
`.env` is git-ignored. Never commit it.

### Run the notebooks in order
Open `notebooks/` in Jupyter or VS Code with the `.venv` kernel and run 01 → 09. Or run headless:
```powershell
foreach ($n in Get-ChildItem notebooks\0*.ipynb | Sort-Object Name) {
  .\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace $n.FullName --ExecutePreprocessor.timeout=3600
}
```
Each notebook reads the files written by the previous ones, so any single notebook can be re-run on its own once its inputs exist. Approximate run times: 01–03 under 1 minute each; 04 about 1 minute; 05 under 1 minute; 06 about 3 minutes; 07 about 5 minutes; 08 about 1 minute; 09 about 10 minutes (it re-runs the test suite and 2,000-sample bootstraps).

### Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

### Explain one customer
```powershell
.\.venv\Scripts\python.exe main.py explain --customer 102 --increase 30000
.\.venv\Scripts\python.exe main.py explain --customer 102 --times-typical-payment 3
.\.venv\Scripts\python.exe main.py explain --customer 102 --share-of-limit 0.25
```
If Neo4j is not reachable, the explanation is printed without the graph-evidence lines.

### Optional: GNN experiment
```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe experiments\gnn_prototype.py
```

### The explanation agent (local LLM, no API key)
An LLM agent that explains credit decisions using evidence from the knowledge graph. **The model and the scoring function make every decision**; the LLM only chooses which tools to call and writes the explanation. The decision line is rendered by code, and the explanation is shown only if a deterministic grounding check passes. See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

It runs on [Ollama](https://ollama.com), locally: no key, no cost, no customer data leaves the machine.
```powershell
winget install --id Ollama.Ollama -e          # or download the installer from ollama.com
ollama pull qwen2.5:7b                         # ~4.7 GB; runs on CPU (slow: tens of seconds per answer)
.\.venv\Scripts\python.exe -m pip install -e .  # makes `python -m credit_kg.agent` work
.\.venv\Scripts\python.exe -m credit_kg.agent ask --customer 2 "Is a 3,000 increase reasonable?" --verbose
.\.venv\Scripts\python.exe -m credit_kg.agent ask "How many customers were late in September 2005?" --verbose
.\.venv\Scripts\python.exe -m credit_kg.agent ask --customer 2 "Explain their risk" --no-cypher   # stage B: fixed tools only
```
| Tool | Wraps |
|---|---|
| `get_customer_profile` | monthly bills, payments, statuses and summary behaviour (protected attributes stripped by code) |
| `get_risk_prediction` | calibrated probability + the five largest per-customer contributions (exact tree-path decomposition of the forest) |
| `score_credit_increase` | the suitability score: score, label, safety checks, points per factor |
| `get_graph_evidence` | the parameterised Neo4j evidence query: late months, streaks, relapses |
| `run_cypher` | guarded read-only Cypher: static checks → `EXPLAIN` must be read-only → read transaction with timeout → row cap → stripping |

#### Evaluating the agent (step 4)
`evals/tasks_v1.jsonl` has 40 tasks whose ground truth is computed by code (`evals/build_tasks.py`): 10 factual lookups, 10 credit decisions, 8 graph/Cypher questions with reference queries, and 12 adversarial tasks (override, protected attributes, writes, Cypher injection, outcome label, nonexistent customers). Each run stores one JSON line per task in `evals/runs/`, and `docs/RESULTS.md` is generated from those runs, never typed by hand.
```powershell
.\.venv\Scripts\python.exe evals\build_tasks.py                              # rebuild the task set (needs Neo4j)
.\.venv\Scripts\python.exe evals\run_eval.py --provider ollama --model qwen2.5:3b   # ~20 min on CPU
.\.venv\Scripts\python.exe scripts\make_results.py                           # regenerate docs/RESULTS.md
```
**Step 5, LLM as a predictor:** can a local LLM predict who misses next month's payment from the same (non-protected) inputs, written as text? It runs on 400 stratified test customers and is compared with the Random Forest, the calibrated probability and logistic regression on the same customers. It also checks determinism at temperature 0 and run-to-run consistency at temperature 0.8.
```powershell
.\.venv\Scripts\python.exe evals\llm_predictor.py --model qwen2.5:3b    # ~600 calls, ~80 min on CPU, $0
```
Results go to `evals/predictor_runs/`, and `scripts/make_results.py` adds them to `docs/RESULTS.md` (section 9).

`evals/heldout/` is reserved for the owner's own questions; the runner refuses to run it unless given `--allow-heldout`, which is used only when the owner asks.

Limits and settings live in `src/credit_kg/agent/settings.py` and can be overridden in `.env` with an `AGENT_` prefix (e.g. `AGENT_MODEL=qwen2.5:3b`, `AGENT_MAX_TOOL_CALLS=6`). Every run is logged as JSON lines in `runs/<run_id>.jsonl`: question, model, each tool call with arguments, latency and result size, guardrail decisions, tokens, estimated cost (always 0 locally), grounding result and final answer.

---

## 9. Project structure

```
graph/
├── dataset/archive/UCI_Credit_Card.csv      raw data (never modified)
├── notebooks/                               one notebook per layer (01-09)
├── src/credit_kg/                           shared logic imported by notebooks
│   ├── config.py          paths, seed, month mapping, behavioural thresholds
│   ├── data.py            load, checks, reshape, aligned coverage, split
│   ├── features.py        baseline features; pandas replica of graph features (parity only)
│   ├── stats.py           Wilson CI, trend test, single-feature AUC
│   ├── viz.py             plotting helpers
│   ├── graph.py           Neo4j driver, batched idempotent loader, integrity counts
│   ├── graph_features.py  Cypher feature extraction, transition probabilities
│   ├── modeling.py        feature sets, RF, repeated CV, thresholds, bootstrap, corrected t-test
│   ├── scoring.py         scenario features, suitability score, explanations
│   ├── evaluation.py      Phase 9: corrected CV intervals, bootstrap CIs, calibration, fairness audit
│   ├── report.py          Phase 9: generates outputs/reports/final_report.md from the results JSON
│   └── agent/             explanation agent: settings, providers (Ollama, mock), tools, guardrails, runner, tracing, CLI
├── queries/
│   ├── schema/constraints.cypher
│   ├── load/01-04_*.cypher                  idempotent MERGE loaders
│   ├── patterns/Q01-Q11_*.cypher            pattern discovery (+ customer_evidence.cypher)
│   └── features/F01-F03_*.cypher            graph feature extraction
├── config/scoring.yaml                      weights, breakpoints, gates, bands, disclaimer
├── tests/                                   pytest suite (data, graph parity, scoring)
├── experiments/gnn_prototype.py             exploratory graph neural network
├── data/processed/, data/features/          generated tables (git-ignored)
├── outputs/figures/, tables/, models/       generated results
├── docs/PLAN.md, docs/FEATURES.md           original plan; generated feature spec
├── docs/Credit_KG_Project_Presentation.pptx 24-slide project presentation (with speaker notes)
├── backups/                                 JSON backup of the Neo4j data that was replaced
├── main.py                                  command-line explanation
├── requirements.txt, .env.example
└── README.md
```

---

## 10. Validation and tests

| Area | How it is checked |
|---|---|
| Month mapping and payment offset | `tests/test_data_and_graph.py`: `status_m6 == PAY_0`; payments match the previous bill 4× more often than the same-month bill |
| No target leakage | target never loaded into Neo4j (checked on stored properties); feature matrices refuse target, split or `scenario_` columns; population statistics from training customers only; thresholds chosen on OOF predictions; test set used once per model |
| Graph correctness | exact node/relationship counts; 6 records per account; `NEXT_MONTH` only t → t+1; 200 sampled records match; loading twice changes nothing |
| Graph features | all 9 equal a pandas re-implementation (notebook 05 and pytest) |
| Model comparison | identical folds, seeds and hyper-parameters; Nadeau–Bengio corrected t-test; 2,000-sample paired bootstrap CI |
| Scenario score (no ground truth) | direction tests on 3,000 real customers; demographic invariance; **scale invariance**; 8 golden personas; edge cases; contributions add up to the score; ±20% weight / ±10% breakpoint stability (Spearman ≥ 0.997, ≤ 3% band changes); default rates by band on the test set with and without the model-risk component (9% → 13% → 45% without it, at a fixed 30,000; the same with the PD safety check also removed) |
| Final evaluation | `tests/test_evaluation.py`: corrected CV interval is wider than the naive one; calibration metrics reward calibrated and catch over-confident predictions; fairness audit table counts and rates. Notebook 09 also re-checks graph counts, outcome-free graph, the test suite and the "not income" wording |

Current status: **32 tests pass**.

---

## 11. Limitations

- **No credit-increase outcomes exist in the data.** The score demonstrates transparent reasoning; it cannot be validated as a lending decision.
- **Six months, one account per customer, no links between customers.** The graph can reorganise and explain behaviour but cannot add information that isn't in these 23 columns. That is why the Random Forest does not improve and all methods converge at ~0.78–0.79 AUC.
- **Status codes are inconsistent across months** (code 1 is rare outside September), so they are used as ordered labels.
- **Projected utilization assumes the bill stays the same**, so it is displayed but not scored.
- **The request ÷ limit breakpoints are a policy assumption**, stated as such in the config.
- **The two-step transition probabilities** for rare state pairs rest on small groups (smoothed).
- **The default model is observational.** Nothing here is causal, and the projected limit is never fed back into the model.
- **Demographics are excluded** from models and the score. Including them adds only +0.0007 AUC.

---

## 12. Glossary

| Term | Meaning here |
|---|---|
| Utilization | `max(bill, 0) / credit_limit` for a month |
| Payment coverage | payment in month t ÷ bill of month t−1, capped at 1; undefined when nothing was due |
| Typical payment | median of the six monthly payments (historical payment behaviour, **not income**) |
| Late month | repayment status ≥ 1 |
| Relapse | late → at least one clean month → late again (a path pattern in the graph) |
| Payback months | hypothetical request ÷ typical payment |
| PD | calibrated probability of default next month, out-of-sample for every customer |
| OOF | out-of-fold: predictions for a training customer from a model that did not see that customer |
