"""Item 1: richer behaviour states + objective motif (pattern) mining.

v1 used 5 repayment states. Here each month gets a richer state:
    repayment group (NO_USE / PAID_FULL / REVOLVING / LATE)
  x card usage band (LOW < 30%, MID 30-90%, HIGH >= 90% of limit)
  x payment band for the payment made this month against LAST month's bill
    (WEAK < 10%, PART 10-99%, FULL >= 99%, NODUE = nothing was due, NA = month 1, no earlier bill)
Late codes 1..9 are merged into LATE because code 1 is recorded inconsistently across months.

Motif mining: every 3-month sequence of (repayment group / payment band) states is a candidate pattern.
A pattern is kept only if (a) enough training customers show it, (b) it is significant after
Benjamini-Hochberg correction, and (c) it stays significant WITHIN groups of customers with the same
number of late months (Mantel-Haenszel test), i.e. it says something the late-month count does not.
"""
import numpy as np
import pandas as pd
from scipy import stats

MONTHS = range(1, 7)


def _repay(st):
    return np.select([st <= -2, st == -1, st == 0], ["NO_USE", "PAID_FULL", "REVOLVING"], "LATE")


def _usage(u):
    return np.select([u < 0.3, u < 0.9], ["LOW", "MID"], "HIGH")


def month_parts(wide: pd.DataFrame):
    """Three (n, 6) arrays: repayment group, usage band, payment band."""
    lim = wide["credit_limit"].to_numpy()
    st = wide[[f"status_m{m}" for m in MONTHS]].to_numpy()
    bill = wide[[f"bill_m{m}" for m in MONTHS]].to_numpy()
    pay = wide[[f"pay_m{m}" for m in MONTHS]].to_numpy()
    repay = _repay(st)
    usage = _usage(np.clip(bill, 0, None) / lim[:, None])
    payb = np.full(st.shape, "NA", dtype=object)
    prev = bill[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(prev > 0, pay[:, 1:] / np.where(prev > 0, prev, 1), np.nan)
    payb[:, 1:] = np.select([prev <= 0, cov < 0.1, cov < 0.99], ["NODUE", "WEAK", "PART"], "FULL")
    return repay, usage, payb


def rich_states(wide: pd.DataFrame) -> np.ndarray:
    r, u, p = month_parts(wide)
    return np.char.add(np.char.add(np.char.add(np.char.add(r.astype(str), "|"), u.astype(str)), "|"), p.astype(str))


def motif_tokens(wide: pd.DataFrame) -> np.ndarray:
    """Per-month token used for motifs: repayment group / payment band (months 2-6 only)."""
    r, _, p = month_parts(wide)
    return np.char.add(np.char.add(r.astype(str), "/"), p.astype(str))[:, 1:]


def customer_motifs(wide: pd.DataFrame) -> pd.DataFrame:
    """Long table: customer_id, motif, start_month (each 3-month window inside months 2-6)."""
    tok = motif_tokens(wide)
    rows = []
    ids = wide["customer_id"].to_numpy()
    for s in range(tok.shape[1] - 2):
        mot = np.char.add(np.char.add(np.char.add(np.char.add(tok[:, s], " > "), tok[:, s + 1]), " > "), tok[:, s + 2])
        rows.append(pd.DataFrame({"customer_id": ids, "motif": mot, "start_month": s + 2}))
    return pd.concat(rows, ignore_index=True)


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def _mantel_haenszel(has, y, strata):
    """MH common odds ratio and continuity-corrected chi-square p-value across strata."""
    num = den = a_sum = e_sum = v_sum = 0.0
    for s in np.unique(strata):
        m = strata == s
        h, yy = has[m], y[m]
        a = np.sum(h & (yy == 1)); b = np.sum(h & (yy == 0))
        c = np.sum(~h & (yy == 1)); d = np.sum(~h & (yy == 0))
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n; den += b * c / n
        n1, n0, m1, m0 = a + b, c + d, a + c, b + d
        a_sum += a; e_sum += n1 * m1 / n
        v_sum += n1 * n0 * m1 * m0 / (n ** 2 * (n - 1)) if n > 1 else 0
    or_mh = num / den if den > 0 else np.inf
    chi = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum if v_sum > 0 else 0
    return or_mh, stats.chi2.sf(chi, 1)


def mine_motifs(wide_train: pd.DataFrame, y_train: pd.Series, late_months: pd.Series,
                min_support: int = 200, fdr: float = 0.05) -> pd.DataFrame:
    cm = customer_motifs(wide_train).drop_duplicates(["customer_id", "motif"])
    ids = wide_train["customer_id"].to_numpy()
    y = pd.Series(np.asarray(y_train), index=ids)
    strata = pd.Series(np.asarray(late_months), index=ids)
    support = cm.groupby("motif")["customer_id"].nunique()
    cand = support[(support >= min_support) & (support <= len(ids) - min_support)].index
    base = y.mean()
    rows = []
    for mot in cand:
        has = pd.Series(False, index=ids)
        has.loc[cm.loc[cm.motif == mot, "customer_id"].to_numpy()] = True
        h = has.to_numpy()
        a, b = np.sum(h & (y == 1)), np.sum(h & (y == 0))
        c, d = np.sum(~h & (y == 1)), np.sum(~h & (y == 0))
        p_raw = stats.chi2_contingency([[a, b], [c, d]])[1]
        or_mh, p_mh = _mantel_haenszel(h, y.to_numpy(), strata.to_numpy())
        rows.append({"motif": mot, "customers": int(h.sum()), "default_rate": a / (a + b),
                     "lift": (a / (a + b)) / base, "p_raw": p_raw, "or_within_late_groups": or_mh, "p_within": p_mh})
    t = pd.DataFrame(rows)
    t["q_raw"], t["q_within"] = _bh(t.p_raw), _bh(t.p_within)
    t["significant"] = t.q_raw < fdr
    t["adds_beyond_late_count"] = t.significant & (t.q_within < fdr)
    return t.sort_values("p_within").reset_index(drop=True)


def transition_late_prob(states: np.ndarray, is_late_next: np.ndarray, alpha: float = 20.0) -> dict:
    """P(late next month | rich state), shrunk towards the overall rate for rare states (alpha pseudo-counts)."""
    s = pd.Series(states[:, :-1].ravel())
    nxt = pd.Series(is_late_next[:, 1:].ravel().astype(float))
    g = nxt.groupby(s).agg(["sum", "size"])
    overall = nxt.mean()
    return ((g["sum"] + alpha * overall) / (g["size"] + alpha)).to_dict() | {"__overall__": overall}


def rich_features(wide: pd.DataFrame, p_late: dict, motif_list: list, risky: set) -> pd.DataFrame:
    """Per-customer features from rich states and selected motifs."""
    st = rich_states(wide)
    ov = p_late["__overall__"]
    p = np.vectorize(lambda x: p_late.get(x, ov))(st)
    f = pd.DataFrame({"customer_id": wide["customer_id"].to_numpy()})
    f["rs_next_late_prob"] = p[:, -1]                       # from the September rich state
    f["rs_mean_late_prob"] = p[:, 1:].mean(axis=1)          # average over the path (months 2-6)
    f["rs_n_distinct_states"] = pd.DataFrame(st).nunique(axis=1).to_numpy()
    cm = customer_motifs(wide)
    has = cm[cm.motif.isin(motif_list)].assign(v=1).pivot_table(index="customer_id", columns="motif", values="v", aggfunc="max")
    has = has.reindex(index=f.customer_id, columns=motif_list).fillna(0)
    f["motif_risky_count"] = has[[m for m in motif_list if m in risky]].sum(axis=1).to_numpy() if risky else 0
    f["motif_protective_count"] = has[[m for m in motif_list if m not in risky]].sum(axis=1).to_numpy()
    return f
