// One MonthlyCreditRecord per customer-month, attached to its account and its repayment state.
UNWIND $rows AS r
MATCH (a:CreditAccount {account_id: r.customer_id})
MATCH (s:RepaymentState {code: r.state})
MERGE (m:MonthlyCreditRecord {record_id: r.record_id})
SET m.customer_id = r.customer_id, m.month_index = r.month_index, m.month = r.month,
    m.bill_amount = r.bill_amount, m.payment_amount = r.payment_amount,
    m.repayment_status = r.repayment_status, m.state = r.state,
    m.utilization = r.utilization, m.is_late = r.is_late, m.credit_balance = r.credit_balance
MERGE (a)-[:HAS_MONTHLY_RECORD]->(m)
MERGE (m)-[:IN_STATE]->(s)
