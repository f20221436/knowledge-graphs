// Q11 Scenario lens (hypothetical): how a requested increase of $increase compares with each customer's
// demonstrated payment behaviour. $increase is a scenario parameter, NOT an observed request.
// typical_payment = median of the 6 monthly payments (mean of the 3rd and 4th sorted values).
// payback_months = months of typical payment needed to repay the full increase if it were drawn.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)
WITH a, m ORDER BY m.payment_amount
WITH a, collect(m.payment_amount) AS pays
WITH a, (pays[2] + pays[3]) / 2.0 AS typical_payment
RETURN a.account_id AS customer_id, a.credit_limit AS credit_limit, typical_payment,
       $increase * 1.0 / a.credit_limit AS increase_to_limit,
       CASE WHEN typical_payment > 0 THEN $increase / typical_payment END AS payback_months
