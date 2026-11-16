// Evidence for one customer's explanation, read from their temporal subgraph.
// weak_payment_months = months whose statement was paid < $weak the following month (coverage on NEXT_MONTH).
MATCH (a:CreditAccount {account_id: $cid})-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)
OPTIONAL MATCH (m)-[r:NEXT_MONTH]->(n:MonthlyCreditRecord)
WITH m, r, n ORDER BY m.month_index
RETURN collect(CASE WHEN m.is_late THEN m.month END) AS late_months,
       collect(CASE WHEN r.coverage IS NOT NULL AND r.coverage < $weak THEN m.month END) AS weak_payment_months,
       collect(CASE WHEN r.status_change = 'improving' THEN n.month END) AS improved_in,
       collect(CASE WHEN r.status_change = 'worsening' THEN n.month END) AS worsened_in
