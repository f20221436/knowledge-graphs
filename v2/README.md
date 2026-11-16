# v2: expected loss, "what would change", and a points scorecard

v2 extends v1 **without modifying any v1 file, notebook, output or the Neo4j graph**. It imports the v1 package (`src/credit_kg`) read-only, reads v1 outputs, and writes only to `v2/outputs/`.

> The credit-increase analysis is a demonstrative scenario, not a lending policy. Requests are hypothetical.

## What's in v2

| Item | Where | What it adds |
|---|---|---|
| 3a. Expected extra loss | `credit_kg_v2/scenario_plus.py`, notebook `v2_01` | Money put at risk by a hypothetical increase = PD × (increase × customer's typical utilization) × LGD |
| 3b. What would change the decision | same | Largest still-supportable request; payment coverage needed to make a request supportable; plain-language levers |
| 4. Points scorecard | `credit_kg_v2/scorecard.py`, notebook `v2_02` | Bank-style scorecard: bands → weight of evidence → logistic regression → points (600 at 50:1 odds, 20 points to double); with vs without graph features |

Settings (LGD, binning, points scaling, feature lists) are in `config_v2.yaml`.

## Results

**Expected extra loss** (request = 3 × each customer's own typical payment):

| Band | Customers | Share of total expected extra loss |
|---|---|---|
| Supportable in this scenario | 18,758 | 41% |
| Review | 1,651 | 9% |
| Not supportable in this scenario | 4,286 | 50% |
| Insufficient behavioural evidence | 5,305 | 0% |

- The same 30,000 for everyone puts **58.5M** of expected extra loss in the "not supportable" band. A relative request (3 × own payment) puts **12.5M** there.

**What would change the decision** (3 × own payment):
- 63.7% of customers have some supportable request; the median is ~20 months of their typical payment.
- Of 11,242 customers in review or not supportable:
  - 8,932 are blocked by a gate;
  - 943 could reach "supportable" by paying more of each bill (median from 5% to ~24%);
  - 1,367 cannot get there by coverage alone.

**Scorecard** (test set, used once):

| Model | Features | ROC-AUC |
|---|---|---|
| v2 scorecard without graph | 10 | 0.7563 |
| v2 scorecard with graph | 15 | 0.7574 |
| v1 logistic regression A / B | 37 / 46 | 0.7573 / 0.7638 |
| v1 Random Forest A | 37 | 0.7800 |

- **Points work as intended:** default rate falls from 53% (lowest-points fifth) to 6% (highest-points fifth).
- **Graph features add no reliable gain inside the compact scorecard** (+0.0011, 95% CI −0.0030 to +0.0053). Banding coarsened them, and two of them collapsed into a single band.

## Assumptions
- **PD:** the probability of default does not change because of the increase.
- **Usage:** new headroom is used at the customer's own typical (median) utilization.
- **LGD = 1.0:** if the customer defaults, everything drawn is lost. This is conservative; the data has no recovery information.

## Run
```powershell
.\.venv\Scripts\python.exe -m pytest v2\tests -q        # 11 tests
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace v2\notebooks\v2_01_expected_loss_and_what_would_change.ipynb v2\notebooks\v2_02_scorecard.ipynb
```

## Item 2: learning from every month (notebook `v2_03`)
Each customer also gives 3 "practice" examples from inside the six months (months 1–3 → late in month 4, …). A model trained on these, without the default label, estimates "late next month" for October.

- **It predicts later months:** trained on earlier windows, it predicts month-6 lateness with AUC 0.786 (the simple rule "late now → late next month" gets 0.780).
- **On its own it predicts default well:** AUC 0.74–0.75.
- **No meaningful gain on top of v1's features:** RF test ΔROC-AUC −0.0013 (CI −0.0037 to +0.0011); LR +0.0001 (CI includes 0). It mostly repackages recent lateness (correlation 0.76).

## Item 1: richer states + objective pattern mining (notebooks `v2_04`, `v2_05`)

**Richer states.** Each month becomes repayment group × usage band × payment band, giving 60 states. They separate risk far better than v1's 5 states: within "late", September default rates range from 23% to 69%.

**Objective pattern mining.** Every 3-month pattern is tested, with false-discovery control (Benjamini–Hochberg) and a second test *within* customers who have the same number of late months (Mantel–Haenszel). Of 53 tested, 45 are significant and **11 add information beyond the late count**. For example:
- weak payments that slide into lateness → odds ×1.9;
- three months of no use with nothing due → ×1.7;
- no new spending while still clearing small bills in full → ×0.61 (safer).

**Effect on models:**

| Model | Test ΔROC-AUC (95% CI) | Verdict |
|---|---|---|
| Random Forest | −0.0017 (−0.0034 to 0.0000) | No gain |
| Logistic regression | **+0.0030 (+0.0002 to +0.0056)** | Small, reliable gain; better in all 25 CV folds |

**v2 graph:** a separate database, never v1's.
- Adds `BehaviourState` nodes (60), `IN_BEHAVIOUR_STATE` links (180k), `Motif` nodes (53) and `SHOWS_MOTIF` links (found by walking `NEXT_MONTH` paths).
- The Cypher logic was checked inside a rolled-back transaction on 300 customers: patterns from graph paths = pandas (900/900), pattern links (153/153) and transition counts all match. The v1 graph was verified unchanged afterwards.
- **To load it fully:**
  1. create a second DBMS (e.g. Neo4j Desktop, port 7688);
  2. copy `v2/.env.v2.example` → `v2/.env.v2` and fill it in;
  3. run notebook `v2_05_graph_v2`. It refuses to run against the v1 database.

## Still open
- **Load and run the v2 graph:** needs the second database (above).
- **Scorecard follow-up:** let the graph and rich-state features keep more bands, then re-test with the full protocol.
