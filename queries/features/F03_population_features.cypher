// F3 Population-based features for every customer, using probabilities learned from training customers:
//   g_markov_next_late_prob: P(late next month | the customer's Aug and Sep states)          ($p2 map)
//   g_path_typicality: average log-probability of the customer's 5 transitions (low = unusual path)  ($logp map)
MATCH (acc:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m5:MonthlyCreditRecord {month_index: 5})-[:NEXT_MONTH]->(m6:MonthlyCreditRecord)
MATCH (acc)-[:HAS_MONTHLY_RECORD]->(:MonthlyCreditRecord)-[r:NEXT_MONTH]->()
WITH acc, m5, m6, avg($logp[r.transition]) AS typicality
RETURN acc.account_id AS customer_id,
       coalesce($p2[m5.state + '|' + m6.state], $fallback) AS g_markov_next_late_prob,
       typicality AS g_path_typicality
