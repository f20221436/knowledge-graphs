// Link every monthly record to its rich behaviour state, and store its motif token (repayment/payment band).
UNWIND $rows AS r
MATCH (m:MonthlyCreditRecord {record_id: r.record_id})
MATCH (s:BehaviourState {code: r.code})
SET m.rs_token = r.token
MERGE (m)-[:IN_BEHAVIOUR_STATE]->(s)
