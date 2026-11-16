// 60 shared rich behaviour states: repayment group | usage band | payment band.
// p_late_next = P(late next month | state), learned from TRAINING customers only.
UNWIND $rows AS r
MERGE (s:BehaviourState {code: r.code})
SET s.repayment = r.repay, s.usage = r.usage, s.payment = r.payband, s.p_late_next = r.p_late_next
