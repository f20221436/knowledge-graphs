// One customer's story: six rich states in order, plus every selected pattern they show with its evidence.
MATCH (a:CreditAccount {account_id: $cid})-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)-[:IN_BEHAVIOUR_STATE]->(s:BehaviourState)
WITH a, m, s ORDER BY m.month_index
WITH a, collect(m.month + ': ' + s.code) AS months
OPTIONAL MATCH (a)-[r:SHOWS_MOTIF]->(p:Motif)
RETURN months, collect(CASE WHEN p IS NULL THEN NULL ELSE
       {pattern: p.pattern, from_month: r.start_month, direction: p.direction,
        odds_ratio: p.odds_ratio_within_late_groups, default_rate_train: p.default_rate_train} END) AS motifs
