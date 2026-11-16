// Q3 High utilization: months where the bill used >= $high_util of the credit limit.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)
WHERE m.utilization >= $high_util
WITH a, count(m) AS high_months, max(m.utilization) AS peak
WHERE high_months >= $min_months
RETURN a.account_id AS customer_id, high_months,
       'utilization >= ' + toString(toInteger($high_util * 100)) + '% in ' + toString(high_months) + ' of 6 months (peak ' + toString(toInteger(peak * 100)) + '%)' AS evidence
