// Schema for the temporal credit-behaviour graph. Safe to run repeatedly (IF NOT EXISTS).
// Uniqueness constraints also create the lookup indexes that make MERGE fast and duplicate-free.
CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE;
CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:CreditAccount) REQUIRE a.account_id IS UNIQUE;
CREATE CONSTRAINT record_id IF NOT EXISTS FOR (m:MonthlyCreditRecord) REQUIRE m.record_id IS UNIQUE;
CREATE CONSTRAINT state_code IF NOT EXISTS FOR (s:RepaymentState) REQUIRE s.code IS UNIQUE;
CREATE INDEX record_month IF NOT EXISTS FOR (m:MonthlyCreditRecord) ON (m.month_index);
CREATE INDEX record_state IF NOT EXISTS FOR (m:MonthlyCreditRecord) ON (m.state);
CREATE INDEX customer_split IF NOT EXISTS FOR (c:Customer) ON (c.split);
