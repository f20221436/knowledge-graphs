# Demo script (about 10 minutes)

Follows the demo flow in `docs/PLAN.md` §19. Every number quoted here comes from `outputs/reports/final_report.md`; if they disagree, the report wins.

**Before you start:** Neo4j running, `.env` filled in, notebooks 01–09 run once.

## 1. The data and one surprising fact (1 min)
- 30,000 card customers in Taiwan, six months each (April–September 2005). The target is whether they missed the next payment (22% did).
- Open `notebooks/01_data_preparation.ipynb`, section on the payment offset: a payment in month *t* matches the bill of month *t − 1* about 4× more often than the same month's bill. So "share of the bill paid" must compare a payment with the **previous** bill.

## 2. Patterns: what held and what didn't (2 min)
- `outputs/reports/final_report.md` §3.
- **Held:** more late months, lower bill coverage, smaller limits and smaller typical payments all go with more missed payments.
- **Didn't hold:** "rising utilization = deterioration" goes the opposite way, and utilization is U-shaped (unused and maxed-out cards are both risky).

## 3. The graph in Neo4j Browser (2 min)
```cypher
MATCH p=(:Customer {customer_id: 2})-[:HAS_ACCOUNT]->()-[:HAS_MONTHLY_RECORD]->
        (:MonthlyCreditRecord {month_index: 1})-[:NEXT_MONTH*5]->()
RETURN p
```
- Six month-nodes in a chain; payment coverage sits on the `NEXT_MONTH` edge because a payment settles the previous month's bill.
- Customer 2 is a **relapse**: late in April, recovered, late again in August. Across all customers, that pattern (Q10) → about 53% miss the next payment on the test set.

## 4. Does the graph help prediction? (2 min)
- `outputs/figures/final_ablation.png` and report §4.
- Random Forest + graph features: **no reliable difference** (test ΔROC-AUC −0.0008, 95% CI [−0.0019, +0.0004]).
- Logistic regression + graph features: **reliable gain** (+0.0065, CI [+0.0027, +0.0101]).
- One-line why: the forest learns streaks from the raw monthly columns; a linear model can't.

## 5. The credit-increase explanation (2 min)
```powershell
.\.venv\Scripts\python.exe main.py explain --customer 2 --times-typical-payment 3
.\.venv\Scripts\python.exe main.py explain --customer 102 --times-typical-payment 3
```
- Customer 2 scores 69 (Review); customer 102 scores 83 (Supportable). The points listed under "supports" and "weighs against" add up exactly to the score.
- Matched real pair (notebook 08, `outputs/tables/scenario_customer_A_vs_B.csv`): same limit and risk, typical payment 20k vs 10k. The lower payer scores lower for the same request, and the gap grows with the request.

## 6. Fairness, limitations and the disclaimer (1 min)
- `outputs/figures/final_fairness.png`. Requests sized to the customer: every group's Supportable rate ≥ 0.83 of the top group's. Fixed 30,000: 21–24-year-olds at 0.48, because their limits are far smaller.
- Read the disclaimer aloud: **demonstrative scenario, not a lending policy; no real increase outcomes; payments are not income.**
