// Q5 Improving repayment: late in Apr-Jun, never late in Jul-Sep, with an observed late -> not-late step.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)
WITH a,
     sum(CASE WHEN m.month_index <= 3 AND m.is_late THEN 1 ELSE 0 END) AS early_late,
     sum(CASE WHEN m.month_index >= 4 AND m.is_late THEN 1 ELSE 0 END) AS recent_late
WHERE early_late > 0 AND recent_late = 0
MATCH (a)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[:NEXT_MONTH]->(y:MonthlyCreditRecord)
WHERE x.is_late AND NOT y.is_late
WITH a, early_late, max(y.month) AS recovered_in
RETURN a.account_id AS customer_id, early_late,
       'late ' + toString(early_late) + 'x in Apr-Jun, recovered in ' + recovered_in + ', clean since' AS evidence
