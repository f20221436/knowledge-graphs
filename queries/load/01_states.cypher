// Five shared repayment-state nodes. Every monthly record links to one of them.
UNWIND $rows AS r
MERGE (s:RepaymentState {code: r.code})
SET s.ordinal = r.ordinal, s.description = r.description
