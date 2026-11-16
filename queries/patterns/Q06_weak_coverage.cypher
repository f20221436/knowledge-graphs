// Q6 Weak payment coverage: NEXT_MONTH edges where the next payment covered < $weak of the statement.
// Edges without a coverage property had nothing due and are ignored.
MATCH (x:MonthlyCreditRecord)-[r:NEXT_MONTH]->(:MonthlyCreditRecord)
WHERE r.coverage IS NOT NULL
WITH x.customer_id AS customer_id, count(r) AS bills_due, sum(CASE WHEN r.coverage < $weak THEN 1 ELSE 0 END) AS weak_edges
WHERE weak_edges >= $min_edges
RETURN customer_id, weak_edges,
       'paid < ' + toString(toInteger($weak * 100)) + '% of the previous statement in ' + toString(weak_edges) + ' of ' + toString(bills_due) + ' months with a bill due' AS evidence
