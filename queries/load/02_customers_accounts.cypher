// One Customer and one CreditAccount per row. No default label is ever stored in the graph.
UNWIND $rows AS r
MERGE (c:Customer {customer_id: r.customer_id})
SET c.age = r.age, c.sex_code = r.sex, c.education_code = r.education,
    c.marriage_code = r.marriage, c.split = r.split
MERGE (a:CreditAccount {account_id: r.customer_id})
SET a.credit_limit = r.credit_limit
MERGE (c)-[:HAS_ACCOUNT]->(a)
