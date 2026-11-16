// Accounts that show a SELECTED motif, found by walking their NEXT_MONTH chain in the graph (months 2-6).
MATCH (p:Motif {selected: true})
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[:NEXT_MONTH]->(y)-[:NEXT_MONTH]->(z)
WHERE x.month_index >= 2 AND x.rs_token + ' > ' + y.rs_token + ' > ' + z.rs_token = p.pattern
MERGE (a)-[r:SHOWS_MOTIF {start_month: x.month_index}]->(p)
RETURN count(r) AS links
