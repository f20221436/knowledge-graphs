# Plan: Knowledge-Graph-Driven Credit Risk & Credit-Increase Scenario Analysis

## ⚠ Revised execution decisions (override the sections below)
- **No Codex.** Claude Code builds everything. Ignore every "CODEX" label; the user reviews at each layer (the "ME" items still apply).
- **One Jupyter notebook per layer**, in `notebooks/`:
  `01_data_preparation`, `02_eda`, `03_pattern_validation_features`, `04_graph_construction`, `05_graph_patterns_features`, `06_baseline_model`, `07_graph_enhanced_model`, `08_credit_increase_scenario`, `09_final_evaluation`.
- **Every code cell has a short markdown cell above it**: 1–3 simple sentences on what the cell does and why.
- A small shared helper package, `src/credit_kg/` (config, paths, Neo4j connection, feature functions, scoring), is imported by the notebooks. This avoids copy-paste between notebooks, lets the pytest checks (alignment, parity, idempotency, scoring properties) run, and keeps the notebook cells short and readable.
- Notebooks pass data to each other through files in `data/processed`, `data/features` and `outputs/`, so each one can be re-run on its own.
- `main.py` shrinks to one command: `explain --customer ID --increase N`. Its logic is also demonstrated in notebook 08.
- Notebooks are executed top to bottom with `jupyter nbconvert --execute` (install `nbformat`, `nbconvert`, `ipykernel` if missing) so they are saved with outputs.

## Context

The workspace (`C:\Users\amartya kumar\graph`) has only the raw dataset, `dataset/archive/UCI_Credit_Card.csv` (30,000 × 25). There is no code yet. The goal is a proof-of-concept that goes from EDA to validated patterns, then to a temporal Neo4j knowledge graph, then to Cypher-derived features. From there it compares a baseline model with a graph-enhanced model for predicting `default.payment.next.month`, and finishes with a clearly hypothetical, explainable credit-increase scenario layer. This document is the architecture and execution plan. Implementation is split between Claude Code (reasoning-heavy work), Codex (mechanical work) and the user (review and understanding).

Environment found: Python 3.14.7 with pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1 and neo4j driver 6.3.1 already installed. Something is listening on `localhost:7687` (probably Neo4j Desktop). `NEO4J_URI` is not set. No git repo.

---

## 1. Executive summary

**What we build:** a pipeline with a CLI (`main.py`) that:
1. Validates and reshapes the data into a wide customer table and a long customer-month table (180k rows).
2. Runs EDA and **statistically validates** behavioral hypotheses before any feature becomes part of the design.
3. Loads a **temporal graph** into Neo4j: Customer → CreditAccount → 6 MonthlyCreditRecord nodes chained by `NEXT_MONTH`. The `NEXT_MONTH` edges carry transition semantics (payment coverage, status change), and shared `RepaymentState` nodes form a population state network.
4. Runs Cypher pattern queries that derive evidence such as "4 of 6 months late" or "late → recovered → late relapse". The target label is **never loaded into the graph**.
5. Extracts **order- and transition-dependent graph features**. Baseline features are order-insensitive aggregates plus latest value and slope.
6. Compares Experiment A (baseline) with Experiment B (baseline + graph features) using the same split, the same model and repeated CV, with a paired bootstrap confidence interval on ΔAUC.
7. Adds Experiment C, a configurable, clearly synthetic **credit-increase suitability score** built from calibrated out-of-fold PD, validated behavioral evidence and *relative* request features, with per-customer factor explanations.

**What the actual data already tells us (the plan depends on these findings):**

| Finding | Evidence | Design consequence |
|---|---|---|
| `PAY_AMTk` pays the **previous** month's bill `BILL_AMT(k+1)` | Exact matches `PAY_AMTk==BILL_AMT(k+1)`: 4,066–4,616 per month vs 582–671 for the same month | Payment coverage = `PAY_AMT_k / BILL_AMT_(k+1)`. That gives **5 aligned pairs**, not 6. In the graph, coverage lives on the `NEXT_MONTH` edge. |
| PAY code `1` almost vanishes after the latest month | Value 1: PAY_0 = 3,688; PAY_2..6 = 28 / 4 / 2 / 0 / 0 | Treat PAY_* as an **ordinal status**, not an exact month counter. Use `late = status ≥ 1`. Treat "severe (≥2)" with care for PAY_0 comparability. |
| Delay is not a strictly cumulative counter | 6,209 of 21,947 late months jump by more than +1 from the prior month | Don't derive "months overdue" arithmetic from the codes. Use state categories. |
| −2 / −1 / 0 are not ordered by risk | PAY_0 default rate: −2 = 13.2%, −1 = 16.8%, 0 = 12.8% | Map to named states (NO_USE / PAID_FULL / REVOLVING), not a numeric "goodness". 25% of PAY_0 = −2 rows still have bills > 0, so −2 ≠ strictly "no consumption". |
| Utilization vs default is **U-shaped** | util_avg ≤ 0: 34.4%; 0.1–0.3: 15.5%; > 1.0: 34.2% | "Higher utilization → more default" holds only above ~0.3. Inactive or credit-balance accounts are **also** risky. The score needs an "insufficient evidence" band and must not reward zero usage. |
| "Rising utilization = deterioration" is **not supported** | Utilization-slope quintiles: falling = 29.8%, mildly rising = 16.1%, strongly rising = 21.3% | Don't hard-code rising utilization as negative. Report this as a rejected hypothesis. |
| Payment coverage is strongly monotone | Median aligned coverage: 0 → 58.1%; ≥ 0.99 → 14.7% | A core scoring signal. |
| Delinquency count is strongly monotone | 0 late months → 11.7% (n = 19,931); 6 late → 70.3% | A core signal. Streak length adds little beyond the count. **Recency** adds signal (1 late month, old-only: 20% vs recent: 34%). |
| Senior's relative-request idea holds | 30,000 / median payment: ≤ 1 → 4.1%; 3–10 → 16.1%; > 100 → 29.3% | Supports the relative framing. Caveat: for a fixed request this is a monotone transform of median payment, so it restates payment capacity rather than adding new evidence. |
| Credit limit and payment size are monotone protective | Limit quintiles 31.8% → 13.8%; median-payment quintiles 29.1% → 13.0% | Keep LIMIT_BAL in the model. The scenario must never feed a *projected* limit back into the model. |
| Performance ceiling is ~0.78 AUC | One quick untuned split: raw cols RF 0.776 / HGB 0.781; aggregates 0.768; + sequence features 0.773; raw + all 0.778 | **Expect graph gain ≈ 0 to +0.005 AUC.** Plan for an honest null result, with CIs. |
| Data quirks | 0 nulls; IDs unique; 35 duplicate feature rows; 3,932 negative bill cells (1,930 rows); 8,274 util > 1 cells (max 10.7); 866 all-zero-bill rows; 1,432 all-zero-payment rows; undocumented EDUCATION 0/5/6 (345 rows) and MARRIAGE 0 (54 rows) | Keep every row. Clip utilization at 0 for credit balances and add a flag. Handle zero or negative denominators explicitly. Group undocumented codes as "other/unknown". |
| Imbalance | 22.12% positive (6,636 / 30,000) | Moderate. Use `class_weight`, stratification, PR-AUC and threshold tuning on OOF predictions. **No SMOTE.** |

**What Neo4j honestly adds:** every per-customer feature on a 6-node chain *can* be computed in pandas, and the plan says so openly. The graph contributes three things:
- (a) A representation where temporal transitions and multi-step path motifs (streaks, relapses, recoveries) are first-class, declarative queries.
- (b) Population-level structure through shared `RepaymentState` nodes: an empirical state-transition network learned from training customers, which yields "next-month late probability" and "trajectory typicality" features.
- (c) Path-level evidence for explanations.

A parity test reproduces graph features in pandas and asserts equality. That guards correctness, and it keeps the claim honest: the experiment tests whether *transition/path/population information* adds value, not whether a database is magic.

---

## 2. Architecture diagram

```
 dataset/archive/UCI_Credit_Card.csv   (raw, read-only)
        │
 [P1] prepare ──► data/processed/customers.parquet (wide, renamed, flags)
        │         data/processed/monthly.parquet   (long: 30k×6, chronological, aligned coverage)
        │         data/processed/split.csv         (fixed stratified 80/20, seed 42)
        ▼
 [P2] EDA ─────► outputs/figures/eda_*.png, outputs/tables/eda_*.csv
 [P3] validate ► outputs/tables/pattern_validation.csv (Wilson CIs) ─► FEATURE SPEC (docs/FEATURES.md)
        │
        ├──► features/baseline.py ─► data/features/baseline.parquet      (order-insensitive aggregates)
        │
 [P4] graph/loader.py ──► Neo4j  (Customer, CreditAccount, MonthlyCreditRecord, RepaymentState)
        │                         NO target label in graph; Customer.split stored
 [P5] queries/patterns/*.cypher ─► pattern membership ─► Python join w/ target ─► pattern default rates
      queries/features/*.cypher ─► data/features/graph.parquet  (+ pandas parity test)
        ▼
 [P6] Exp A: baseline  ─┐  same split, same RF config, 5×5 repeated CV + test, OOF-calibrated PD
 [P7] Exp B: + graph   ─┴► outputs/tables/model_comparison.csv, paired bootstrap ΔAUC CI
        ▼
 [P8] Exp C: scenario (requested_increase ∈ {10k,20k,30k,50k,custom}) ─► relative features
        + calibrated OOF PD + behavioral evidence ─► suitability score (config/scoring.yaml)
        ─► `python main.py explain --customer 102 --increase 30000`
        ▼
 [P9] outputs/reports/final_report.md (results, null results, limitations)
```

---

## 3. Detailed 9-phase workflow

Notation: **CLAUDE** = needs reasoning; **CODEX** = can be delegated; **ME** = the user must understand or verify.

### Phase 1: Data Collection & Preparation (≈1.5 h)
- **Objective:** a trusted, documented, reshaped dataset and a single fixed train/test split.
- **Inputs:** `UCI_Credit_Card.csv`.
- **Outputs:** `customers.parquet`, `monthly.parquet`, `split.csv`, `docs/DATA_DICTIONARY.md`, `outputs/tables/data_quality.csv`.
- **Tasks (30-min blocks):**
  1. Scaffold the repo, config, `.env.example` and requirements (CODEX, 30m).
  2. Loader with schema checks: 25 columns, ints, ID unique, target ∈ {0,1}; data-quality report (CODEX, 30m).
  3. Month mapping and long reshape. Suffix 1 = Sep 2005 = `month_index 6`; suffix 6 = Apr 2005 = `month_index 1`. PAY_0 ↔ suffix 1. Add `prev_bill` and aligned `coverage` (CLAUDE writes the spec, CODEX implements, 30m).
  4. Stratified split (seed 42, test 20%) saved once and reused by everything (CODEX, 15m).
- **Design decisions:** rename to `status_m1..m6`, `bill_m1..m6`, `pay_m1..m6` (chronological) so month order is never ambiguous. Keep raw PAY codes plus a `repayment_state` enum: −2 NO_USE, −1 PAID_FULL, 0 REVOLVING, 1 LATE_1, ≥2 LATE_2PLUS. Keep duplicates and negative bills. Map EDUCATION {0,5,6} → 4 "other/unknown" and MARRIAGE 0 → 3 "other".
- **Dependencies:** none.
- **Could go wrong:** off-by-one month mapping (the #1 risk); misaligning PAY_AMT with the same-month bill.
- **Validation:** unit test that `PAY_AMT1 == BILL_AMT2` match counts reproduce ~4,066; test that `month_index 6` status equals raw PAY_0; row counts 30,000 / 180,000; split class ratios within ±0.5%.
- **ME:** understand the month mapping and the payment-to-previous-bill offset. It will come up in every explanation.

### Phase 2: EDA & Pattern Recognition (≈2 h)
- **Objective:** find candidate relationships without assuming them.
- **Inputs:** `customers.parquet`, `monthly.parquet` (**train split only** for anything that drives design; full data only for descriptive counts).
- **Outputs:** ~12 figures and tables; `notebooks/01_eda.ipynb` with written interpretation.
- **Tasks:**
  1. Distributions of limit, bill, payment and utilization (log scales); outlier table for util > 1, negative bills and zero rows (CODEX, 30m).
  2. Default rate by bucket for utilization (avg / latest / max), coverage, late count, recency, max status, limit, median payment and slopes; plus class balance (CODEX makes plots from the CLAUDE-specified bucket list, 45m).
  3. Status transition heatmap (month t → t+1) and monthly default-rate trajectories by late group (CODEX, 30m).
  4. Interaction tables: late_count × recency, util band × coverage band, limit × util (CODEX, 15m).
  5. Interpretation write-up that separates **observed association** from engineered rule and from model relationship (CLAUDE, 30m).
- **Design decisions:** Spearman correlations for skewed amounts; no causal language; demographics are descriptive only.
- **Could go wrong:** reading the U-shaped utilization curve as monotone; heavy tails wrecking plots.
- **Validation:** bucket sizes shown next to every rate; no bucket under ~100 is interpreted.
- **ME:** be able to state three patterns that held and two that didn't (rising utilization, −2/−1/0 ordering).

### Phase 3: Pattern Validation & Feature Engineering (≈2.5 h)
- **Objective:** turn EDA hunches into validated or rejected claims, then freeze the feature specification.
- **Tasks:**
  1. For each hypothesis H1–H8 (utilization, delinquency, coverage, payment trend, bill trend, consistency, limit, deterioration), compute grouped default rates with **Wilson 95% CIs**, a χ² or Cochran-Armitage trend test, and a single-feature AUC on train. Record the verdict as Supported / Non-monotone / Not supported (CODEX runs the harness, CLAUDE interprets, 60m).
  2. Redundancy pass: Spearman |ρ| > 0.9 clusters, keeping one per cluster (e.g., util_avg / util_max / util_latest are ρ ≈ 0.92–0.95; late_count / late_streak / late_recent3 are ≈ 0.9) (CLAUDE, 30m).
  3. Implement `features/baseline.py` from the frozen spec (§5) (CODEX, 45m).
  4. Unit tests for division-by-zero, negative bills, all-zero rows and chronological slopes (CODEX, 15m).
- **Design decisions:** baseline = **order-insensitive** aggregates (mean, median, max, min, count, std) plus latest value and OLS slope. Anything that needs **adjacency** (transitions, streaks, relapses) is reserved for the graph phase, which keeps the A-vs-B split principled.
- **Could go wrong:** tuning bucket thresholds on the full data (leakage into design). Use train only.
- **ME:** review `outputs/tables/pattern_validation.csv`. Each feature in the spec must point to a validation row or a stated reason.

### Phase 4: Knowledge Graph Design & Construction (≈2.5 h)
- **Objective:** an idempotent, constrained, temporal graph (schema in §7).
- **Tasks:**
  1. Confirm the Neo4j instance, version and database; `.env` (ME + CODEX, 15m).
  2. `graph/connection.py` (driver from env vars, `verify_connectivity`) (CODEX, 15m).
  3. `queries/schema/constraints.cypher` (CLAUDE designs, CODEX writes, 15m).
  4. Batched `UNWIND` + `MERGE` loader, 2,000–5,000 customers per transaction, in 5 passes: states → customers + accounts → monthly records → `HAS_MONTHLY_RECORD` → `NEXT_MONTH` + `IN_STATE` (CODEX, 60m).
  5. Post-load integrity queries and the idempotency test (CLAUDE specifies, CODEX implements, 30m).
- **Design decisions:** see §7. Load ~240k nodes and ~720k relationships. Expect 1–3 minutes locally.
- **Could go wrong:** `MERGE` on unindexed properties (very slow), which is why constraints are created first; `MERGE` on whole patterns creating duplicates, which is why the loader merges nodes by key and then relationships; driver version or connection issues.
- **Validation:** exact counts (30,000 / 30,000 / 180,000 / 5 states; 150,000 `NEXT_MONTH`); no account with ≠ 6 records; no `NEXT_MONTH` where `b.month_index ≠ a.month_index + 1`; counts unchanged after a second run.
- **ME:** be able to draw the schema and justify each edge.

### Phase 5: Pattern Analysis & Feature Extraction (≈2.5 h)
- **Objective:** Cypher discovers behavioral evidence; Python validates it against the target outside the graph; graph features are extracted.
- **Tasks:**
  1. Write pattern queries Q1–Q11 (§8) (CLAUDE, 60m).
  2. Query runner: load `.cypher`, pass parameters, return a DataFrame (CODEX, 30m).
  3. Pattern validation: pattern member IDs → join target in Python → default rate + Wilson CI vs the 22.1% base rate (CODEX, 15m).
  4. Feature extraction queries → `graph.parquet`; the transition model is computed **over train customers only** (CLAUDE query logic, CODEX plumbing, 30m).
  5. **Parity test:** reimplement each graph feature in pandas and assert equality (CODEX, 30m).
- **Could go wrong:** population features leaking test customers; variable-length paths exploding (bounded at ≤ 5 hops, so it's fine); Cypher `percentileCont` / null semantics.
- **ME:** run 2–3 queries by hand in Neo4j Browser and confirm one customer's result matches the raw CSV row.

### Phase 6: Baseline ML Model (≈1.5 h)
- See §9. Tasks: training / evaluation module (CODEX, 45m); methodology review, threshold and calibration logic (CLAUDE, 30m); results review (ME, 15m).

### Phase 7: Graph-Enhanced ML Model (≈1.5 h)
- See §10. Tasks: reuse the Phase 6 pipeline with a feature-set switch (CODEX, 20m); repeated CV plus paired bootstrap comparison (CODEX, 30m); ablations and importance of graph features (CLAUDE, 30m); honest write-up (CLAUDE + ME, 15m).

### Phase 8: Credit-Increase Score & Decision Explanation (≈3 h)
- See §6, §11 and §12. Tasks: scenario features (CODEX from spec, 30m); scoring engine with YAML config (CODEX, 45m); scoring design and breakpoints anchored to Phase 3 tables (CLAUDE, 45m); explanation generator and CLI (CODEX, 30m); sensitivity analysis plus the "Customer A vs B" real-pair finder (CLAUDE specifies, CODEX builds, 30m).

### Phase 9: Final Evaluation (≈1.5–2 h)
- Final report: data findings, validated and rejected patterns, graph schema, A vs B with CIs, scenario sensitivity, limitations (CLAUDE drafts, 60m); README and demo script (CODEX, 30m); ME rehearses the demo (30m).

---

## 4. Dataset-to-feature mapping

| Raw column | Meaning (after inspection) | Used as |
|---|---|---|
| ID | customer key | `customer_id`, graph key; never a feature |
| LIMIT_BAL | current credit limit (NT$) | `credit_limit` (baseline feature; utilization denominator; scenario base) |
| SEX, EDUCATION, MARRIAGE, AGE | demographics | Graph properties for description only. **Excluded** from primary models and from the score. Reported in a separate ablation. |
| PAY_0, PAY_2..PAY_6 | repayment status Sep → Apr (ordinal codes) | `status_m6..m1`, `repayment_state` → delinquency features, `IN_STATE` edges |
| BILL_AMT1..6 | statement balance Sep → Apr (can be negative) | `bill_m6..m1` → utilization and bill features |
| PAY_AMT1..6 | amount paid in Sep → Apr, **paying the previous month's statement** | `pay_m6..m1` → payment features; coverage on the aligned edge. **Never "income".** |
| default.payment.next.month | Oct 2005 default | **Target only.** Not in the graph, not in any feature. |
| — | scenario | `requested_increase`: hypothetical config parameter, never an observation |

---

## 5. Feature engineering specification (frozen after Phase 3)

Conventions: m1 = Apr … m6 = Sep. `util_t = max(bill_t, 0) / limit`. `coverage_t = min(pay_t / bill_{t-1}, 1.0)` for t = 2..6 when `bill_{t-1} > 0`, otherwise NaN, with the month counted as "no balance due". The slope is the OLS slope over month_index.

| Group | Keep (baseline) | Drop / why |
|---|---|---|
| A. Utilization | `util_avg`, `util_latest`, `util_std`, `util_slope`, `high_util_months` (≥ 0.9), `inactive_flag` (all util ≤ 0.02), `credit_balance_months` (bill < 0) | `util_max`, `util_median`: ρ ≥ 0.92 with util_avg |
| B. Coverage | `coverage_median`, `coverage_latest`, `coverage_min`, `weak_cov_months` (< 0.1), `full_pay_months` (≥ 0.99), `no_balance_months` | `coverage_std`, `coverage_slope`: 5 points and noisy; test in P3, keep only if validated |
| C. Delinquency | `late_months` (status ≥ 1), `late_recent3` (m4–m6), `max_status`, `status_latest` (m6 raw code, one-hot of state), `paid_full_months` (−1), `no_use_months` (−2) | `avg_delay` (codes aren't additive); `severe_delay_count` merged into `max_status`; streaks → **graph** |
| D. Bills | `bill_latest`, `bill_slope_rel` (slope / limit) | `bill_avg`, `bill_max`: collinear with utilization × limit |
| E. Payments | `pay_median` (= **typical demonstrated payment**), `pay_latest`, `pay_slope_rel`, `zero_pay_months` | `pay_max`, `pay_avg`: skewed by one-off lump sums; the median is robust |
| F. Stability | `pay_cv` (std / mean, 0 if mean = 0), `util_std` (above) | "behavioral volatility" composite: hard to explain |
| Raw | `credit_limit` + the 18 raw monthly columns (strong baseline; raw columns alone reach ~0.78 AUC) | — |

**Graph-derived features** (Phase 5, `g_` prefix; all need adjacency or population structure):
`g_max_late_streak`, `g_late_onsets` (not-late → late edges), `g_recoveries` (late → not-late), `g_relapses` (late → not-late → late 3-hop motif), `g_latest_transition` (worsening / stable / improving on edge m5 → m6), `g_max_util_rise_run` (longest run of consecutive utilization increases > 0.05), `g_weak_cov_after_late` (weak coverage on the edge leaving a late month), `g_markov_next_late_prob` (P(late at t+1 | state at m6) from the train-population transition network), `g_path_typicality` (mean log P of the customer's 5 transitions under that network).

**Leakage rules:** nothing uses the target except evaluation; transition probabilities are fit on train customers; thresholds (e.g., 0.9, 0.1) are fixed from P3 train tables before modeling; no future (Oct) information exists in the features.

---

## 6. Relative credit-increase feature specification (Experiment C only)

`R` = `requested_increase`: a **scenario parameter** (default 30,000; sweep 10k / 20k / 30k / 50k / custom). These features are stored in `scenario_*` columns and never mixed into the ML training features.

| Feature | Formula | Status |
|---|---|---|
| `increase_to_limit` | R / limit | **Scored** |
| `payback_months` (= increase_to_typical_payment) | R / pay_median. If pay_median = 0, set to NaN and raise the flag `no_demonstrated_payment`. | **Scored**. Read as "months of the customer's typical demonstrated payment to repay the full increase if drawn". Not income. |
| `typical_payment_to_increase` | pay_median / R | Shown only (the inverse of the above; redundant) |
| `increase_to_recent_payment` | R / median(pay_m4..m6) | Explanation only, shown when it differs from payback_months by > 50% (trend signal) |
| `projected_limit` | limit + R | Display |
| `current_util` | max(bill_m6, 0) / limit | Scored via utilization context (same as `util_latest`) |
| `projected_util` | max(bill_m6, 0) / (limit + R) | Display. **Mechanically ≤ current_util** under the static-bill assumption, so it isn't evidence and isn't rewarded. |
| `util_change_after_increase` | projected_util − current_util | Display only (mechanical) |
| `current_bill_to_limit`, `projected_bill_to_new_limit` | — | **Dropped**: duplicates of current_util / projected_util |

Rule: projected limit is **never** fed back into the default model to obtain a "post-increase PD". The model has no causal basis for that.

---

## 7. Knowledge graph schema

```
(:Customer {customer_id, age, sex_code, education_code, marriage_code, split})
   -[:HAS_ACCOUNT]->
(:CreditAccount {account_id, credit_limit})
   -[:HAS_MONTHLY_RECORD]->
(:MonthlyCreditRecord {record_id:"<cid>-<m>", customer_id, month_index 1..6, month:"2005-04".."2005-09",
                       bill_amount, payment_amount, repayment_status (raw code), state, utilization,
                       is_late, credit_balance})
   -[:NEXT_MONTH {coverage (pay_t / bill_{t-1}, capped 1, null if no balance), util_delta,
                  status_change (worsening|stable|improving), transition:"REVOLVING->LATE_2PLUS"}]->
(:MonthlyCreditRecord)
(:MonthlyCreditRecord)-[:IN_STATE]->(:RepaymentState {code: NO_USE|PAID_FULL|REVOLVING|LATE_1|LATE_2PLUS, ordinal})
```
- **Why coverage is on `NEXT_MONTH`:** a payment in month t settles the statement of month t−1, so it is a property of the transition, not of either month. This is a data-driven reason for the edge to exist.
- **Why `RepaymentState` nodes:** 180k records attach to 5 shared nodes. All `NEXT_MONTH` edges between train customers aggregate into a population state-transition network, which is the one genuinely cross-customer structure the data supports.
- **Demographic nodes:** **not created**. They would add hub nodes with no reasoning value and invite demographic "patterns". Kept as Customer properties for descriptive queries.
- **No target label in the graph.** `Customer.split` is stored (not sensitive) so that population queries can filter to train customers.
- **Constraints:** `UNIQUE` on Customer.customer_id, CreditAccount.account_id, MonthlyCreditRecord.record_id, RepaymentState.code. **Indexes:** MonthlyCreditRecord.month_index, MonthlyCreditRecord.state, Customer.split.
- **Idempotency:** `MERGE` on the unique key, then `SET` properties; relationships `MERGE`d between already-matched endpoints. Running twice gives identical counts, and a test asserts it.
- **Temporal representation:** integer `month_index` for ordering plus an ISO month label. `NEXT_MONTH` only links index t → t+1 of the same account.
- **Optional (polished only):** `(:CreditAccount)-[:SIMILAR_BEHAVIOR {score}]->(:CreditAccount)` kNN over train behavior vectors (needs the GDS plugin). Cut first.

---

## 8. Cypher analysis plan (`queries/patterns/`)

None of these queries touches a default label. Each returns `customer_id` plus its evidence.

| # | Query | Graph logic | Example evidence returned |
|---|---|---|---|
| Q1 | Repeated delinquency | count `IN_STATE` → LATE_* per account ≥ 3 | "4 of 6 months late" |
| Q2 | Longest late streak | variable-length `NEXT_MONTH*1..5` path where `all(n IN nodes(p) WHERE n.is_late)` | "late Jun→Sep (4 consecutive)" |
| Q3 | High utilization | records with utilization ≥ 0.9, count ≥ 3 | "util ≥ 90% in 5 months" |
| Q4 | Rising utilization run | consecutive edges with util_delta > 0.05 | "util rose 3 months running" (reported as *not* risk-increasing per P3) |
| Q5 | Improving repayment | late in m1–m3, no late in m4–m6, ≥ 1 `improving` edge | "recovered in Jul, clean since" |
| Q6 | Weak payment coverage | `NEXT_MONTH.coverage < 0.1` count ≥ 3 | "paid < 10% of prior statement in 4 of 5 months" |
| Q7 | Stable payer | all non-null coverage ≥ 0.99 and no late | "paid prior statement in full every month" |
| Q8 | Multi-signal concern | ≥ 2 of {Q1, Q3, Q6, recent late} | list of signals triggered |
| Q9 | Population transition network | `(a)-[:IN_STATE]->(s1), (a)-[:NEXT_MONTH]->(b)-[:IN_STATE]->(s2)` over `split='train'` → counts / probabilities | 5×5 transition matrix |
| Q10 | Relapse motif | `(x)-[:NEXT_MONTH]->(y)-[:NEXT_MONTH*1..3]->(z)` with late → not-late → late | "recovered in May, relapsed in Aug" |
| Q11 | Scenario lens (param `$increase`) | latest record + median payment → payback months, increase/limit; finds pairs with similar limit and PD but very different payback | "Customer A vs B" demo pair |

Validation (Python): each Q's member set → default rate + Wilson CI vs the 22.1% base rate → `outputs/tables/graph_pattern_validation.csv`.

---

## 9. Baseline ML methodology (Experiment A)

- **Target:** `default.payment.next.month`. **Features:** §5 baseline + raw monthly columns + limit. **No demographics** (ablation A+demo reported separately). A0 = raw columns only, as a reference.
- **Model:** `RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features='sqrt', class_weight='balanced_subsample', random_state=42)`. Light tuning of min_samples_leaf / max_depth with 5-fold CV on train, identical for A and B. Logistic regression (standardized) as an interpretable sanity check.
- **Protocol:** fixed 80/20 stratified split from P1. 5×5 repeated stratified CV on train for model selection and comparison. Test set touched **once** per final model.
- **Imbalance:** class weights. Decision threshold chosen on **OOF** predictions to maximize F1, and also reported at 0.5.
- **Metrics:** ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix; accuracy for reference only.
- **Calibration:** `CalibratedClassifierCV(isotonic, cv=5)` for the PD used in scoring. Reliability plot and Brier score.
- **PD for scenario:** OOF PD for train customers and test-model PD for test customers, so no customer is scored by a model that saw its label.
- **Importance:** permutation importance on test (not impurity importance, which is biased toward continuous features).

## 10. Graph-enhanced ML methodology (Experiment B)

- B = A's features + the 9 `g_*` features pulled from Neo4j (loaded from `graph.parquet`, produced by Cypher). Same seeds, folds, hyperparameters and threshold procedure.
- **Proof the graph is used:** the feature matrix builder raises an error if any `g_*` column is missing or constant, and logs the column list plus source-query hashes into `outputs/models/*_meta.json`.
- **Comparison:** per-fold paired ΔAUC and ΔPR-AUC (mean ± sd over 25 folds); test-set paired bootstrap (2,000 resamples) 95% CI for ΔAUC. Claim an "improvement" only if the CI excludes 0.
- **Ablations:** B-minus-population (drop the Markov features) and B-graph-only (g_* + limit) show *which* graph information matters. Permutation importance of g_* inside B.
- **Expectation, stated in advance:** Δ ≈ 0–0.005 AUC (see the §1 sanity check). A null result is reported as a finding: "temporal-transition structure is largely captured by strong raw monthly columns."

## 11. Credit-increase scenario methodology (Experiment C)

Clearly labeled everywhere: **"Demonstrative scenario framework, not a banking underwriting policy. The dataset has no credit-increase outcomes."**

Score = 100 × Σ wᵢ·sᵢ, with sᵢ ∈ [0,1] from piecewise-linear maps. Weights and breakpoints live in `config/scoring.yaml`, and breakpoints are **anchored to Phase 3 validated bucket tables**:

| Component | Input | Default weight | Map (initial) |
|---|---|---|---|
| Model risk | calibrated PD | 0.30 | PD ≤ 0.10 → 1; ≥ 0.60 → 0 |
| Repayment record | late_months, status_latest | 0.20 | 1 − late_months/6; ×0.5 if latest month late |
| Payment coverage | coverage_median | 0.15 | 0 → 0; 0.1 → 0.4; 0.5 → 0.8; ≥ 0.99 → 1 |
| Request vs demonstrated payment | payback_months | 0.20 | ≤ 3 → 1; 12 → 0.5; ≥ 36 → 0; no payment → 0 |
| Request vs limit | increase_to_limit | 0.10 | ≤ 0.25 → 1; 1.0 → 0.3; ≥ 2 → 0 |
| Utilization context | util_latest (U-shape aware) | 0.05 | 0.1–0.5 → 1; > 0.9 → 0.3; inactive → 0.5 (neutral) |

- **Gates** (override the band): PD ≥ 0.5 or latest status ≥ 2 → "Not supportable in scenario"; `inactive_flag` or `no_demonstrated_payment` → "Insufficient behavioral evidence".
- **Bands:** ≥ 70 "Supportable in this scenario", 50–70 "Review", < 50 "Not supportable in this scenario".
- **Double counting:** PD already includes delinquency. This is kept deliberately at a modest weight for transparency and stated openly.
- **Analyses:**
  1. Sweep R over 10k / 20k / 30k / 50k: band distribution shift.
  2. Weight perturbation ±20% → Spearman rank stability.
  3. **A vs B real pair:** two actual customers with similar limit and PD but typical payments of ~20k vs ~10k, both at R = 30k, showing different payback months and scores.
  4. Sanity only: actual default rate by band on test (should decrease). This is **not** evidence of "suitability".

### 11a. How we validate what has no label

Only the **scenario score** lacks ground truth. The default model has a real label (`default.payment.next.month`, 6,636 positives). Graph patterns are checked against that label *outside* the graph, and graph correctness is checked against the raw CSV. For the score, "correct" cannot be proven, so it is validated as a **well-behaved, consistent, traceable instrument**:

| Check | What it proves | How (test file) |
|---|---|---|
| **Metamorphic / monotonicity tests** | The logic moves the right way | For every customer: raising R never raises the score; raising PD, late_months or payback_months never raises it; raising coverage never lowers it (`tests/test_scoring.py`, property-based over real rows) |
| **Invariance tests** | No hidden drivers | Changing sex, age, education or marriage leaves the score exactly unchanged |
| **Golden personas** | Agreed expectations | ~8 hand-built customers (clean payer + small request, 6-late, inactive, zero payments, credit balance, util > 1, …) with the expected band **written by ME before running**; the test asserts the band |
| **Edge-case tests** | No crashes or infinities | Zero payment → flag, not ∞; negative bill; all-zero rows |
| **Traceability** | The explanation is honest | Σ contributions + 50 == score; every displayed value equals the stored feature column (3 random customers) |
| **Proxy check with the real label (non-circular variant)** | Behavioral components carry real risk information | On **test** customers: actual default rate by band for (a) the full score and (b) the score **with the PD component removed**. (b) is the non-circular check. Necessary, not sufficient: default ≠ suitability. |
| **Stability** | Not an artifact of arbitrary weights | Weights ±20% and breakpoints ±10%: Spearman rank ρ and % band flips; target ρ > 0.9 |
| **Face validity review** | A human agrees it makes sense | ME reads 15 explanations sampled across bands and scenario amounts and logs disagreements |
| **Held-out discipline** | No tuning to the checks | Breakpoints set from P3 **train** tables; proxy check run on test once |

**Explicitly not validatable (stated in the report):** whether granting an increase leads to good outcomes, because the dataset has no increase decisions or post-increase behavior. The score is a demonstration of transparent reasoning, not an evaluated underwriting model.

## 12. Explainability approach

- **Score layer:** each component's signed contribution = wᵢ·(sᵢ − 0.5)·100 → ranked "supports / weighs against" list. Every number traces to a named feature column.
- **ML layer:** global permutation importance; per-customer, show the top model inputs as the customer's value and population percentile. (Polished: SHAP TreeExplainer, if `shap` installs on Python 3.14; otherwise skip.)
- **Graph evidence:** attach Cypher path evidence (e.g., Q2/Q10 month ranges) for that customer.
- **CLI output** (`python main.py explain --customer 102 --increase 30000`):
```
Customer 102 | SCENARIO: hypothetical increase 30,000 (not an observed request)
Current limit 100,000 → projected 130,000
Historical evidence (Apr–Sep 2005): avg util 0.42 | latest util 0.55 | median coverage 0.18
  late months 1/6 (Jun) | typical demonstrated payment 4,000 | util trend +0.02/mo
Graph evidence: no late streak >1; recovered Jul; next-month late prob (population) 0.11
Scenario-relative: increase/limit 0.30 | payback ≈ 7.5 months of typical payment | projected util 0.42 (mechanical)
Model: default probability 0.14 (calibrated, out-of-fold)
Suitability score 68/100 → REVIEW (demonstrative framework)
 + low predicted default risk (+11)   + request small vs limit (+4)
 − weak payment coverage (−5)         − payback >6 months (−2)
```

---

#

## 14. Time/effort estimates

| Phase | Blocks | Hours | Depends on |
|---|---|---|---|
| 1 Prep | 30+30+30+15 (+15 review) | 2.0 | — |
| 2 EDA | 30+45+30+15+30 | 2.5 | 1 |
| 3 Validation + features | 60+30+45+15 (+15 review) | 2.75 | 2 |
| 4 Graph build | 15+15+15+60+30 (+15 debug) | 2.5 | 1 (spec from 3 for properties) |
| 5 Cypher + graph features | 60+30+15+30+30 | 2.75 | 3, 4 |
| 6 Baseline | 45+30+15 | 1.5 | 3 |
| 7 Graph-enhanced | 20+30+30+15 | 1.6 | 5, 6 |
| 8 Scenario + explain | 30+45+45+30+30 | 3.0 | 6 (7 optional) |
| 9 Final | 60+30+30 | 2.0 | all |
| **Total** | | **≈20.5 h** (≈16 h without buffers or polish) | |

Phase 4 can run in parallel with Phase 3 once the P1 long table exists.

## 15. MVP plan (~8 h)
1. P1 prep: reshape, alignment, split (1.25h).
2. P2+P3 condensed into one script: 6 bucket tables with Wilson CIs, no notebook (1.5h).
3. P4 loader without state nodes: Customer / Account / Monthly / `NEXT_MONTH` with coverage on the edge (1.5h).
4. P5: Q1, Q2, Q6, Q10 plus 5 g_* features (streak, onsets, recoveries, relapses, latest transition) (1.25h).
5. P6+P7: RF A vs B on the single test split + bootstrap CI (1h).
6. P8: fixed-weight score, R = 30k plus sweep, text `explain` CLI (1.25h).
7. README with results and disclaimers (0.25h).

## 16. Solid (~10–14 h) and Polished (~15–20 h) versions
- **Solid:** MVP + `RepaymentState` nodes and the train-only transition network (Markov features) + full Q1–Q11 + 5×5 repeated CV + calibration + ablations + A-vs-B real pair + sensitivity sweep + EDA notebook.
- **Polished:** + SHAP, GDS kNN `SIMILAR_BEHAVIOR` (with train-only neighbor features), logistic-regression scorecard comparison, fairness ablation table, Neo4j Bloom/Browser screenshots, final report with figures, pytest suite in CI.
- **Cut first, in order:** GDS similarity → SHAP → notebooks (keep scripts) → repeated CV (keep bootstrap) → weight-perturbation analysis → Q4/Q7 → logistic sanity model. **Never cut:** month alignment tests, leakage rules, idempotency test, A-vs-B CI, scenario disclaimers.

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Target leakage | Target not in graph; single split file; population stats train-only; `assert` no target-derived columns in the feature matrix |
| Future info | Only Apr–Sep inputs exist; Oct appears only as target; scenario variables are kept out of ML |
| PAY_* misread | State enum; ordinal treatment; unit test mapping PAY_0 → m6; documented code-1 anomaly |
| Payment/bill misalignment | Offset alignment with a test reproducing the ~4k exact-match evidence |
| PAY_AMT as income | Terminology lint: grep outputs and docs for "income" / "salary" in the P9 checklist |
| Arbitrary score logic | Breakpoints cite P3 tables; config file; sensitivity analysis; disclaimers |
| Decorative graph | g_* features only from Cypher; parity test; ablations; honest statement of what's pandas-equivalent |
| Graph features not actually used | Feature-builder asserts g_* present and non-constant; metadata logs columns |
| Duplicate nodes | Constraints before load; `MERGE` by key; run-twice test |
| Wrong temporal order | `month_index` + integrity query on `NEXT_MONTH` |
| Class imbalance | Balanced weights, stratification, PR-AUC, OOF threshold |
| Overfitting / noisy comparison | min_samples_leaf, repeated CV, paired bootstrap CI, test touched once |
| Redundant features | Spearman-cluster pruning in P3 |
| Demographic concerns | Excluded from primary models and the score; separate ablation only; no demographic nodes |
| Score mistaken for policy | Banner in CLI, README and report; "in this scenario" band wording |
| U-shaped utilization mis-scored | Inactive → neutral / insufficient evidence; no reward for zero usage |
| Mechanical projected utilization "rewarding" increases | Display only; not scored |
| Python 3.14 package gaps (e.g., shap) | Only already-installed libraries are core; shap optional |

## 18. Validation checklist
- [ ] 30,000 customers / 180,000 monthly rows; m6 status == PAY_0
- [ ] Coverage uses previous-month bill; alignment test passes
- [ ] Split file reused by every phase; class ratio matches
- [ ] Every kept feature → validation row or reason in `docs/FEATURES.md`
- [ ] Neo4j counts exact; second load changes nothing; `NEXT_MONTH` index + 1 only
- [ ] No target property anywhere in the graph (`MATCH (n) WHERE n.default IS NOT NULL` returns 0)
- [ ] Graph feature parity test passes
- [ ] Transition network computed on train only
- [ ] A and B share folds, seeds and hyperparameters; ΔAUC CI reported
- [ ] Threshold chosen on OOF, not test
- [ ] Scenario features absent from the ML matrices
- [ ] Explanation numbers equal the stored feature values (test on 3 customers)
- [ ] Score monotonicity, invariance, golden-persona and edge-case tests pass (§11a)
- [ ] Band default-rate check run with and without the PD component (test set, once)
- [ ] Disclaimer present in CLI, README and report; no "income" wording

## 19. Final demo flow (~10 min)
1. Data: 6 months per customer; the payment-pays-previous-bill discovery (1 min).
2. Patterns: validated (delinquency, coverage, limit) vs rejected (rising utilization, U-shaped utilization) (2 min).
3. Neo4j Browser: one customer's 6-node chain, coverage on edges, the state network, Q10 relapse result (2 min).
4. A vs B table with CI and the honest verdict (2 min).
5. `explain` for the real Customer A vs B pair at R = 30k, then a sweep to 50k (2 min).
6. Limitations and the synthetic-score disclaimer (1 min).

## 20. Recommended implementation order & project structure

```
graph/
  dataset/archive/UCI_Credit_Card.csv     (raw, untouched)
  config/settings.yaml  config/scoring.yaml
  src/credit_kg/
    config.py
    data/{load.py, reshape.py, split.py}
    features/{baseline.py, scenario.py}
    eda/{plots.py, validation.py}
    graph/{connection.py, loader.py, runner.py, extract.py}
    models/{train.py, evaluate.py, compare.py}
    scoring/{suitability.py, explain.py}
  queries/{schema/, load/, patterns/Q01..Q11.cypher, features/*.cypher}
  notebooks/{01_eda.ipynb, 02_pattern_validation.ipynb}
  data/{processed/, features/}   outputs/{figures/, tables/, models/, reports/}
  tests/{test_reshape.py, test_features.py, test_graph_idempotency.py, test_parity.py, test_scoring.py}
  docs/{PLAN.md, DATA_DICTIONARY.md, FEATURES.md}
  main.py  (subcommands: prepare | eda | validate | load-graph | analyze | extract | train --exp A|B | compare | scenario --increase N | explain --customer ID --increase N | report)
  requirements.txt  .env.example  README.md
```
Order:
1. Copy this plan to `docs/PLAN.md` and `git init`.
2. Phase 1 → 2 → 3, with Phase 4 in parallel after Phase 1.
3. Phase 5.
4. Phase 6.
5. Phase 7.
6. Phase 8.
7. Phase 9.

Each step ends with its tests plus a ME review before the next CLAUDE design step.

## Verification (end-to-end)
`python main.py prepare && python main.py validate && python main.py load-graph && python main.py load-graph` (the second run must not change counts) `&& python main.py extract && python main.py train --exp A && python main.py train --exp B && python main.py compare && python main.py explain --customer 102 --increase 30000`, then `pytest -q`, then walk the §18 checklist.

**Assumptions to confirm at approval:** the local Neo4j on :7687 (version? GDS plugin?) is the target DB; demographics are excluded from primary models; scripts first, notebooks for narrative only.
