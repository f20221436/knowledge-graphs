// Q9 Population state-transition network: how training customers move between the shared RepaymentState
// nodes from one month to the next. Test customers are excluded so later features cannot leak.
MATCH (c:Customer {split: 'train'})-[:HAS_ACCOUNT]->(:CreditAccount)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[:NEXT_MONTH]->(y:MonthlyCreditRecord)
MATCH (x)-[:IN_STATE]->(s1:RepaymentState), (y)-[:IN_STATE]->(s2:RepaymentState)
RETURN s1.code AS from_state, s2.code AS to_state, count(*) AS transitions
