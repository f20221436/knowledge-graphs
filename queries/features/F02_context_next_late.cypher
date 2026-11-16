// F2 Second-order population statistics (training customers only): for every pair of consecutive states
// (month t, month t+1), how often was the customer late in month t+2? Three-node NEXT_MONTH paths.
MATCH (c:Customer {split: 'train'})-[:HAS_ACCOUNT]->(:CreditAccount)-[:HAS_MONTHLY_RECORD]->(a:MonthlyCreditRecord)
      -[:NEXT_MONTH]->(b:MonthlyCreditRecord)-[:NEXT_MONTH]->(z:MonthlyCreditRecord)
RETURN a.state + '|' + b.state AS context, sum(CASE WHEN z.is_late THEN 1 ELSE 0 END) AS next_late, count(*) AS n
