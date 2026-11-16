// Every 3-month pattern each customer shows, read straight from the graph's NEXT_MONTH paths (for parity with pandas).
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[:NEXT_MONTH]->(y)-[:NEXT_MONTH]->(z)
WHERE x.month_index >= 2
RETURN a.account_id AS customer_id, x.month_index AS start_month,
       x.rs_token + ' > ' + y.rs_token + ' > ' + z.rs_token AS motif
