// Population movement between rich states (training customers only): how often each state is followed by a late month.
MATCH (c:Customer {split: 'train'})-[:HAS_ACCOUNT]->(:CreditAccount)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[:NEXT_MONTH]->(y:MonthlyCreditRecord)
MATCH (x)-[:IN_BEHAVIOUR_STATE]->(s:BehaviourState)
RETURN s.code AS state, sum(CASE WHEN y.is_late THEN 1 ELSE 0 END) AS late_next, count(*) AS n
