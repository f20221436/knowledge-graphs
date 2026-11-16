// One node per mined 3-month pattern, with its training-data evidence. selected = adds information beyond late count.
UNWIND $rows AS r
MERGE (p:Motif {pattern: r.motif})
SET p.customers_train = r.customers, p.default_rate_train = r.default_rate, p.odds_ratio_within_late_groups = r.or_within,
    p.q_value_within = r.q_within, p.selected = r.selected, p.direction = r.direction
