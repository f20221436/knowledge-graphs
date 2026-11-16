// F1 Per-customer transition features read from the NEXT_MONTH edges:
//   onsets (not late -> late), recoveries (late -> not late), weak coverage right after a late month,
//   and the direction of the most recent transition (Aug -> Sep): +1 worsening, 0 stable, -1 improving.
MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(x:MonthlyCreditRecord)-[r:NEXT_MONTH]->(y:MonthlyCreditRecord)
RETURN a.account_id AS customer_id,
       sum(CASE WHEN NOT x.is_late AND y.is_late THEN 1 ELSE 0 END) AS g_late_onsets,
       sum(CASE WHEN x.is_late AND NOT y.is_late THEN 1 ELSE 0 END) AS g_recoveries,
       sum(CASE WHEN x.is_late AND r.coverage IS NOT NULL AND r.coverage < $weak THEN 1 ELSE 0 END) AS g_weak_cov_after_late,
       max(CASE WHEN y.month_index = 6 THEN
             CASE r.status_change WHEN 'worsening' THEN 1 WHEN 'improving' THEN -1 ELSE 0 END END) AS g_latest_transition
