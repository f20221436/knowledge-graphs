// Q7 Stable full payer: every statement that was due got paid in full the next month, >= $min_edges times, never late.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[r:NEXT_MONTH]->(y:MonthlyCreditRecord)
WITH a, collect(r.coverage) AS cov, max(CASE WHEN x.is_late OR y.is_late THEN 1 ELSE 0 END) AS any_late
WITH a, cov, any_late, size([c IN cov WHERE c IS NOT NULL]) AS due
WHERE any_late = 0 AND due >= $min_edges AND all(c IN cov WHERE c IS NULL OR c >= $full)
RETURN a.account_id AS customer_id, due AS full_payments,
       'paid the previous statement in full in all ' + toString(due) + ' months with a bill due, never late' AS evidence
