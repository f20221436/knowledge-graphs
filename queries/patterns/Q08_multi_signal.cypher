// Q8 Multiple concerning signals at once (>= $min_signals of: repeated lateness, high utilization,
// weak coverage, late in the latest month). Each signal is derived from the graph, not from a label.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)
OPTIONAL MATCH (m)-[r:NEXT_MONTH]->()
WITH a,
     sum(CASE WHEN m.is_late THEN 1 ELSE 0 END) AS late_months,
     sum(CASE WHEN m.utilization >= $high_util THEN 1 ELSE 0 END) AS high_util_months,
     sum(CASE WHEN r.coverage IS NOT NULL AND r.coverage < $weak THEN 1 ELSE 0 END) AS weak_cov_edges,
     max(CASE WHEN m.month_index = 6 AND m.is_late THEN 1 ELSE 0 END) AS late_latest
WITH a, [x IN [
          CASE WHEN late_months >= 3 THEN 'repeated lateness (' + toString(late_months) + '/6)' END,
          CASE WHEN high_util_months >= 3 THEN 'high utilization (' + toString(high_util_months) + '/6)' END,
          CASE WHEN weak_cov_edges >= 3 THEN 'weak payment coverage (' + toString(weak_cov_edges) + '/5)' END,
          CASE WHEN late_latest = 1 THEN 'late in latest month' END] WHERE x IS NOT NULL] AS signals
WHERE size(signals) >= $min_signals
RETURN a.account_id AS customer_id, size(signals) AS n_signals,
       reduce(s = '', x IN signals | s + CASE s WHEN '' THEN '' ELSE '; ' END + x) AS evidence
