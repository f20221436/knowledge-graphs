// Q10 Relapse motif: late -> recovered (not late for 1-4 months) -> late again.
// A pure path pattern: recovery edge (x->y), a run of non-late months (path p = y..w), then a new onset (w->z).
// p is matched separately so that the "all months not late" check covers only the recovered run.
MATCH (x:MonthlyCreditRecord {is_late: true})-[:NEXT_MONTH]->(y:MonthlyCreditRecord {is_late: false})
MATCH p = (y)-[:NEXT_MONTH*0..3]->(w:MonthlyCreditRecord)
MATCH (w)-[:NEXT_MONTH]->(z:MonthlyCreditRecord {is_late: true})
WHERE all(n IN nodes(p) WHERE NOT n.is_late)
WITH x.customer_id AS customer_id, y, z ORDER BY y.month_index
WITH customer_id, count(*) AS relapses, collect('recovered ' + y.month + ', late again ' + z.month) AS episodes
WHERE relapses >= $min_relapses
RETURN customer_id, relapses, reduce(s = '', e IN episodes | s + CASE s WHEN '' THEN '' ELSE '; ' END + e) AS evidence
