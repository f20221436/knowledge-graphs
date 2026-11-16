// Q4 Rising utilization run: consecutive NEXT_MONTH edges where utilization rose by more than $rise.
// Maximal runs only. (Notebook 03 found rising utilization is NOT associated with more default.)
MATCH (a:MonthlyCreditRecord)-[r0:NEXT_MONTH]->()
WHERE r0.util_delta > $rise
  AND NOT EXISTS { MATCH ()-[rp:NEXT_MONTH]->(a) WHERE rp.util_delta > $rise }
MATCH p = (a)-[:NEXT_MONTH*1..5]->(b)
WHERE all(r IN relationships(p) WHERE r.util_delta > $rise)
  AND NOT EXISTS { MATCH (b)-[rn:NEXT_MONTH]->() WHERE rn.util_delta > $rise }
WITH a.customer_id AS customer_id, length(p) AS rises, a.month AS from_month, b.month AS to_month
ORDER BY rises DESC
WITH customer_id, collect({rises: rises, from_month: from_month, to_month: to_month})[0] AS best
WHERE best.rises >= $min_rises
RETURN customer_id, best.rises AS rising_months,
       'utilization rose ' + toString(best.rises) + ' months running (' + best.from_month + ' to ' + best.to_month + ')' AS evidence
