"""Step 5: parsing, prompt contents, sampling, summary metrics and the RESULTS.md section. No model calls."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_kg import config as C
from credit_kg import llm_predictor as P
from credit_kg import modeling as M

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text,expected", [('{"probability": 0.3}', 0.3), ('{"probability": 35}', 0.35), ("0.22", 0.22),
                                           ('{"probability": -0.1}', None), ('{"probability": 150}', None), ("no idea", None)])
def test_parse_probability(text, expected):
    got = P.parse_probability(text)
    assert got == pytest.approx(expected) if expected is not None else got is None


@pytest.fixture(scope="module")
def table():
    return M.load_table()


def test_prompt_has_no_protected_attributes_or_outcome(table):
    feats = M.feature_sets()["A"]
    row = table.iloc[0]
    text = P.make_prompt(row, feats).lower()
    for word in ["sex", "gender", "age", "education", "marriage", "marital", "default"]:
        assert f"{word}:" not in text and f" {word} " not in text
    with pytest.raises(AssertionError):
        P.make_prompt(row, feats + ["sex"])
    with pytest.raises(AssertionError):
        P.make_prompt(row, feats + [C.TARGET])


def test_sample_is_stratified_test_only_and_reproducible(table):
    s = P.build_sample(table, 400)
    assert len(s) == 400 and (s.split == "test").all() and s.customer_id.is_unique
    test_rate = table[table.split == "test"][C.TARGET].mean()
    assert s[C.TARGET].sum() == round(400 * test_rate)
    assert P.build_sample(table, 400).customer_id.tolist() == s.customer_id.tolist()


def test_summary_metrics(table):
    sample = P.build_sample(table, 60)
    y = sample.set_index("customer_id")[C.TARGET]
    rng = np.random.default_rng(0)
    good = (y * 0.5 + rng.uniform(0, 0.5, len(y))).clip(0, 1)
    recs = [{"customer_id": int(c), "phase": "main", "run": 0, "temperature": 0.0, "probability": float(good[c]) if i else None,
             "latency_s": 2.0, "prompt_tokens": 600, "completion_tokens": 10} for i, c in enumerate(y.index)]
    recs += [{**recs[1], "phase": "determinism"}]
    for run in (1, 2, 3):
        recs += [{**r, "phase": "consistency", "run": run, "temperature": 0.8,
                  "probability": round(float(good[r["customer_id"]]), 1)} for r in recs[1:11]]
    base = {"Calibrated PD (RF)": pd.Series(rng.uniform(0, 1, len(y)), index=y.index)}
    s = P.summarise(recs, sample, base)
    assert s["parsed"] == 59 and s["coverage"] == pytest.approx(59 / 60)
    assert s["models"]["LLM"]["roc_auc"] > 0.9 and "LLM - Calibrated PD (RF)" in s["paired_differences"]
    assert s["determinism_temp0"]["identical_share"] == 1.0
    assert s["consistency"]["runs"] == 3 and s["consistency"]["all_runs_identical_share"] == 1.0
    assert s["tokens_per_call"] == 610


def _mr():
    spec = importlib.util.spec_from_file_location("make_results", ROOT / "scripts" / "make_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pred(ci_hi):
    return {"run_id": "p1", "model": "m", "code_version": "x", "customers": 400, "positives": 88, "parsed": 400, "coverage": 1.0,
            "distinct_llm_values": 9, "llm_mean_probability": 0.3, "observed_rate": 0.22, "started": "2026-11-12T03:00:00",
            "models": {"LLM": {"roc_auc": 0.6, "roc_ci": [0.55, 0.65], "brier": 0.2},
                       "Calibrated PD (RF)": {"roc_auc": 0.78, "roc_ci": [0.73, 0.83], "brier": 0.14}},
            "paired_differences": {"LLM - Calibrated PD (RF)": {"delta_roc_auc": -0.18, "ci": [-0.25, ci_hi]}},
            "latency_mean_s": 8.0, "latency_p50_s": 8.0, "latency_p95_s": 9.0, "tokens_per_call": 686, "cost_usd": 0.0, "calls_total": 600}


def test_results_step5_section_and_claim():
    mr = _mr()
    md = mr.build_markdown([], None, _pred(-0.1))
    assert "## 9. Step 5" in md and "LLM reliably worse" in md and "Tested an LLM as a credit-risk predictor" in md
    assert "Step 5: LLM-as-predictor" not in md.split("## 3.")[0]          # removed from "not yet run"
    md2 = mr.build_markdown([], None, _pred(+0.02))
    assert "Tested an LLM as a credit-risk predictor" not in md2 and "no reliable difference" in md2


def test_latest_predictor_ignores_pilots(tmp_path):
    import json
    (tmp_path / "a_summary.json").write_text(json.dumps({**_pred(-0.1), "customers": 20, "started": "2026-11-12T05:00:00"}))
    (tmp_path / "b_summary.json").write_text(json.dumps(_pred(-0.1)))
    assert _mr().latest_predictor(tmp_path)["customers"] == 400
