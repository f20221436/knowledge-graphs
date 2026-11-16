// Q1 Repeated delinquency: customers whose monthly records point to a LATE_* state in >= $min_late months.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)-[:IN_STATE]->(s:RepaymentState)
WHERE s.code STARTS WITH 'LATE'
WITH a, m ORDER BY m.month_index
WITH a, count(m) AS late_months, collect(m.month) AS late_in
WHERE late_months >= $min_late
RETURN a.account_id AS customer_id, late_months,
       toString(late_months) + ' of 6 months late (' + reduce(s = '', x IN late_in | s + CASE s WHEN '' THEN '' ELSE ', ' END + x) + ')' AS evidence
