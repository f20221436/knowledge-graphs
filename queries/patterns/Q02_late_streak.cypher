// Q2 Longest consecutive late streak: walk NEXT_MONTH paths whose every month is late.
// Only maximal runs are considered (the month before the start and after the end are not late).
MATCH (start:MonthlyCreditRecord {is_late: true})
WHERE NOT EXISTS { MATCH (prev:MonthlyCreditRecord)-[:NEXT_MONTH]->(start) WHERE prev.is_late }
MATCH p = (start)-[:NEXT_MONTH*0..5]->(finish:MonthlyCreditRecord)
WHERE all(n IN nodes(p) WHERE n.is_late)
  AND NOT EXISTS { MATCH (finish)-[:NEXT_MONTH]->(nxt:MonthlyCreditRecord) WHERE nxt.is_late }
WITH start.customer_id AS customer_id, length(p) + 1 AS streak, start.month AS from_month, finish.month AS to_month
ORDER BY streak DESC
WITH customer_id, collect({streak: streak, from_month: from_month, to_month: to_month})[0] AS best
WHERE best.streak >= $min_streak
RETURN customer_id, best.streak AS longest_streak,
       'late ' + toString(best.streak) + ' months in a row (' + best.from_month + ' to ' + best.to_month + ')' AS evidence
