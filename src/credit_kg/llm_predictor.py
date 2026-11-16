"""Step 5: can an LLM predict who misses next month's payment as well as the trained models?

The LLM sees the same non-protected inputs as model A, written out as plain text, and returns a probability.
It is compared with the Random Forest, the calibrated probability and logistic regression on the same customers.
The outcome is used only to score the predictions afterwards, never in a prompt.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config as C
from . import modeling as M
from .agent.tools import feature_meaning

INSTRUCTIONS = ("You are estimating credit risk from six months of card history (April-September 2005). "
                "Given the customer's data below, output ONLY a JSON object {\"probability\": p} where p is the probability "
                "(0 to 1) that the customer misses next month's payment.\n\nCustomer data:\n")
# A pilot on 20 customers with "About 22% of customers miss it." added gave 0.22 for almost everyone (2 distinct values):
# the 3B model copied the hint instead of judging the customer. The hint was removed before the main run.
JSON_SCHEMA = {"type": "object", "properties": {"probability": {"type": "number"}}, "required": ["probability"]}


def build_sample(table: pd.DataFrame, n: int = 400, seed: int = C.SEED) -> pd.DataFrame:
    """Stratified sample of test customers: the same share who missed a payment as the whole test set."""
    te = table[table.split == "test"]
    k = round(n * te[C.TARGET].mean())
    pos = te[te[C.TARGET] == 1].sample(k, random_state=seed)
    neg = te[te[C.TARGET] == 0].sample(n - k, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)      # shuffled, so any prefix is mixed


def make_prompt(row: pd.Series, features: list) -> str:
    assert not set(C.DEMOGRAPHICS) & set(features) and C.TARGET not in features, "protected or outcome column in prompt"
    return INSTRUCTIONS + "\n".join(f"- {feature_meaning(f)}: {row[f]:g}" for f in features)


def parse_probability(text: str):
    """The model's probability, or None if the reply is not a usable number in [0, 1]. Percentages are accepted."""
    try:
        p = float(json.loads(text)["probability"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        m = re.search(r"-?\d+(?:\.\d+)?", text or "")
        if not m:
            return None
        p = float(m.group())
    if 1 < p <= 100:
        p /= 100
    return p if 0 <= p <= 1 else None


def _boot(y, preds: dict, n_boot=2000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    idx = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if y[i].min() != y[i].max()]
    out = {}
    for name, p in preds.items():
        aucs = np.array([roc_auc_score(y[i], p[i]) for i in idx])
        out[name] = {"roc_auc": float(roc_auc_score(y, p)), "roc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
                     "brier": float(brier_score_loss(y, p)), "_boot": aucs}
    return out


def summarise(records: list, sample: pd.DataFrame, baselines: dict) -> dict:
    """records: one dict per LLM call (customer_id, phase, run, probability or None, latency_s, tokens).
    baselines: {name: Series of probabilities indexed by customer_id} for the same customers."""
    main = pd.DataFrame([r for r in records if r["phase"] == "main"]).set_index("customer_id")
    y_all = sample.set_index("customer_id")[C.TARGET]
    ok = main["probability"].notna()
    ids = main.index[ok]
    y = y_all.loc[ids].to_numpy()
    preds = {"LLM": main.loc[ids, "probability"].to_numpy(dtype=float), **{k: v.loc[ids].to_numpy(dtype=float) for k, v in baselines.items()}}
    boot = _boot(y, preds)
    diffs = {f"LLM - {k}": {"delta_roc_auc": boot["LLM"]["roc_auc"] - boot[k]["roc_auc"],
                            "ci": [float(np.percentile(boot["LLM"]["_boot"] - boot[k]["_boot"], 2.5)),
                                   float(np.percentile(boot["LLM"]["_boot"] - boot[k]["_boot"], 97.5))]} for k in baselines}
    for v in boot.values():
        v.pop("_boot")
    lat = main["latency_s"].to_numpy()
    res = {"customers": int(len(main)), "positives": int(y_all.loc[main.index].sum()), "parsed": int(ok.sum()),
           "coverage": float(ok.mean()), "distinct_llm_values": int(main["probability"].nunique()),
           "llm_mean_probability": float(main["probability"].mean()), "observed_rate": float(y.mean()),
           "models": boot, "paired_differences": diffs,
           "latency_mean_s": float(lat.mean()), "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
           "calls_over_60s": int(sum(r["latency_s"] > 60 for r in records)),          # e.g. the laptop sleeping mid-call
           "top_llm_values": [[float(v), int(n)] for v, n in main["probability"].value_counts().head(5).items()],
           "tokens_per_call": float((main["prompt_tokens"] + main["completion_tokens"]).mean()), "cost_usd": 0.0,
           "calls_total": len(records)}

    det = [r for r in records if r["phase"] == "determinism"]
    if det:
        first = main["probability"].to_dict()
        same = [r["probability"] == first.get(r["customer_id"]) for r in det]
        res["determinism_temp0"] = {"customers": len(det), "identical_share": float(np.mean(same))}

    cons = pd.DataFrame([r for r in records if r["phase"] == "consistency"])
    if not cons.empty:
        wide = cons.pivot_table(index="customer_id", columns="run", values="probability", aggfunc="first").dropna()
        runs = list(wide.columns)
        pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
        yc = y_all.loc[wide.index].to_numpy()
        res["consistency"] = {
            "temperature": float(cons["temperature"].iloc[0]), "customers": int(len(wide)), "runs": len(runs),
            "mean_abs_difference": float(np.mean([np.abs(wide[a] - wide[b]).mean() for a, b in pairs])),
            "all_runs_identical_share": float((wide.nunique(axis=1) == 1).mean()),
            "mean_spearman_between_runs": float(np.mean([wide[a].corr(wide[b], method="spearman") for a, b in pairs])),
            "roc_auc_per_run": [float(roc_auc_score(yc, wide[r])) if 0 < yc.sum() < len(yc) else None for r in runs]}
    return res


def baselines_for(sample: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Test-set probabilities of the trained models for the sampled customers (all out-of-sample)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ids = sample.customer_id.to_numpy()
    rf = pd.read_parquet(C.MODELS / "test_pred_A.parquet").set_index("customer_id")["p_A"]
    pdc = pd.read_parquet(C.MODELS / "pd_calibrated.parquet").set_index("customer_id")["pd"]
    X, y, _, _, _, _ = M.matrices(table, "A")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, y)
    t = table.set_index("customer_id").loc[ids]
    lr_p = pd.Series(lr.predict_proba(t[M.feature_sets()["A"]])[:, 1], index=ids)
    return {"Random Forest (raw)": rf.loc[ids], "Calibrated PD (RF)": pdc.loc[ids], "Logistic regression": lr_p}
