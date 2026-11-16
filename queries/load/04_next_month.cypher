// Temporal chain month t -> t+1. The edge carries what happens *between* the months:
// coverage = payment made in month t+1 / bill of month t (absent when nothing was due),
// the utilization change and the direction of the repayment-status change.
UNWIND $rows AS r
MATCH (a:MonthlyCreditRecord {record_id: r.from_id})
MATCH (b:MonthlyCreditRecord {record_id: r.to_id})
MERGE (a)-[n:NEXT_MONTH]->(b)
SET n.coverage = r.coverage, n.util_delta = r.util_delta,
    n.status_change = r.status_change, n.transition = r.transition
