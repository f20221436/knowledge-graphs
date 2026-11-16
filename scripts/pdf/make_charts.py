"""Charts + numbers for the plain-language credit-increase PDF. Run with the project venv from the project root."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from credit_kg import scoring as S, config as C

OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 220, "savefig.bbox": "tight",
})

t = S.load_scoring_inputs()
cfg = S.load_config()
comp = cfg["components"]
inc3 = S.relative_increase(t, "payment_multiple", 3)
res = S.score(t, inc3, cfg)
d = pd.concat([t, res.drop(columns="customer_id")], axis=1)
facts = {}

SHORT = {"model_risk": "Predicted risk of missing\nnext payment",
         "repayment": "Late-payment record",
         "coverage": "Share of last bill paid",
         "payback": "Months of usual payment\nto cover the request",
         "request_vs_limit": "Request size vs\ncurrent limit",
         "utilization": "How much of the limit\nis in use now"}


# 1. Contribution charts for the worked examples
def contrib_chart(cid, fname):
    r = d.loc[d.customer_id == cid].iloc[0]
    vals = {k: float(r[f"contrib_{k}"]) for k in comp}
    order = sorted(vals, key=lambda k: vals[k])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(order))
    v = [vals[k] for k in order]
    ax.barh(y, v, color=[BLUE if x >= 0 else RED for x in v], height=0.6)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [SHORT[k].replace("\n", " ") for k in order], fontsize=9, color=INK2)
    lim = max(16, max(abs(x) for x in v) + 3)
    ax.set_xlim(-lim, lim)
    for yi, x in zip(y, v):
        ax.text(x + (0.4 if x >= 0 else -0.4), yi, f"{x:+.1f}", va="center",
                ha="left" if x >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlabel("Points added to (or taken from) the neutral starting score of 50", fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / fname)
    plt.close(fig)
    facts[f"c{cid}"] = {**{f"pts_{k}": round(vals[k], 1) for k in comp}, "score": float(r.score), "band": r.band,
                        "gate": r.gate, "pd": float(r.pd), "limit": float(r.credit_limit),
                        "pay_median": float(r.pay_median), "request": float(r.scenario_requested_increase),
                        "late": int(r.late_months), "status": int(r.status_latest),
                        "coverage": float(r.coverage_median), "util": float(r.util_latest),
                        "payback": float(r.scenario_payback_months), "to_limit": float(r.scenario_increase_to_limit),
                        **{f"sub_{k}": float(r[f"sub_{k}"]) for k in comp}}


contrib_chart(102, "contrib_102.png")
contrib_chart(2, "contrib_2.png")
for cid in (1825, 10):
    r = d.loc[d.customer_id == cid].iloc[0]
    facts[f"c{cid}"] = {"score": float(r.score), "band": r.band, "gate": r.gate, "pd": float(r.pd),
                        "late": int(r.late_months), "status": int(r.status_latest),
                        "pay_median": float(r.pay_median), "limit": float(r.credit_limit)}

# 2. Small multiples: how each factor turns into points
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
panels = [
    ("model_risk", np.linspace(0, 0.8, 200), "Predicted chance of missing next payment", lambda x: f"{x:.0%}"),
    ("repayment", None, "Late months out of 6", None),
    ("coverage", np.linspace(0, 1, 200), "Share of previous bill paid (typical month)", lambda x: f"{x:.0%}"),
    ("payback", np.linspace(0, 40, 200), "Request ÷ usual monthly payment (months)", None),
    ("request_vs_limit", np.linspace(0, 2.5, 200), "Request ÷ current limit", lambda x: f"{x:.0%}"),
    ("utilization", np.linspace(0, 1.2, 200), "Share of limit currently used", lambda x: f"{x:.0%}"),
]
from matplotlib.ticker import FuncFormatter
for ax, (k, xs, xl, fmt) in zip(axes.flat, panels):
    w = comp[k]["weight"]
    if k == "repayment":
        lm = np.arange(7)
        ax.plot(lm, 1 - lm / 6, color=BLUE, lw=2, marker="o", ms=4, label="latest month on time")
        ax.plot(lm, (1 - lm / 6) * 0.5, color=ORANGE, lw=2, marker="o", ms=4, label="latest month late")
        ax.text(2.6, 0.78, "latest month on time", fontsize=7, color=INK2)
        ax.text(0.1, 0.2, "latest month late", fontsize=7, color=INK2)
    else:
        ys = np.interp(xs, comp[k]["x"], comp[k]["y"])
        ax.plot(xs, ys, color=BLUE, lw=2)
        if fmt:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _, f=fmt: f(x)))
    ax.set_title(f"{SHORT[k]}\nweight {w:.0%}", fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel(xl, fontsize=7.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1], ["0", "half", "full"], fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
fig.supylabel("Points earned for this factor", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "factor_curves.png")
plt.close(fig)

# 3. Real default rate by label, held-out test customers
te = d[d.split == "test"]
order = [S.BAND_SUPPORT, S.BAND_REVIEW, S.BAND_NOT, S.BAND_INSUFFICIENT]
nice = ["Supportable", "Review", "Not supportable", "Insufficient evidence"]
g = te.groupby("band")["default_next_month"].agg(["mean", "size"]).reindex(order)
base = te.default_next_month.mean()
fig, ax = plt.subplots(figsize=(6.2, 2.9))
x = np.arange(4)
ax.bar(x, g["mean"], color=BLUE, width=0.55)
ax.axhline(base, color=INK2, lw=1, ls="--")
ax.set_xlim(-0.5, 4.0)
ax.text(3.33, base + 0.012, f"average {base:.0%}", ha="left", fontsize=8.5, color=INK2)
for xi, m in zip(x, g["mean"]):
    ax.text(xi, m + 0.015, f"{m:.0%}", ha="center", fontsize=10, color=INK, weight="bold")
ax.set_xticks(x, [f"{lab}\n{n:,} customers" for lab, n in zip(nice, g["size"])], fontsize=8.5, color=INK2)
ax.set_ylim(0, 0.75)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("Missed the next payment", fontsize=9)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0, pad=4)
fig.savefig(OUT / "band_default.png")
plt.close(fig)
facts["band_default"] = {b: [float(m), int(n)] for b, (m, n) in zip(order, g.itertuples(index=False))}
facts["base_test"] = float(base)
facts["n_test"] = int(len(te))

# 4. Fixed vs own-scale request across limit thirds
third = pd.qcut(d.credit_limit, 3, labels=["Low-limit third", "Middle third", "High-limit third"])
shares = {}
for name, amt in [("Fixed 30,000 for everyone", 30000), ("3 × the customer's own usual payment", inc3)]:
    rr = S.score(t, amt, cfg)
    shares[name] = (rr.band == S.BAND_SUPPORT).groupby(third.values).mean().reindex(third.cat.categories)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
x = np.arange(3)
wbar = 0.34
for i, (name, s) in enumerate(shares.items()):
    xs_ = x + (i - 0.5) * (wbar + 0.03)
    ax.bar(xs_, s.values, width=wbar, color=[BLUE, ORANGE][i], label=name)
    for xi, v in zip(xs_, s.values):
        ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, list(third.cat.categories), fontsize=9, color=INK2)
ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("Labelled “Supportable”", fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
fig.savefig(OUT / "fixed_vs_relative.png")
plt.close(fig)
facts["shares"] = {k: [round(float(v), 3) for v in s.values] for k, s in shares.items()}

# 5. Matched pair
mp = pd.read_csv(C.ROOT / "outputs" / "tables" / "scenario_customer_A_vs_B.csv") if hasattr(C, "ROOT") else \
    pd.read_csv("outputs/tables/scenario_customer_A_vs_B.csv")
facts["pair"] = mp.to_dict("records")

(OUT / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
print(json.dumps(facts, indent=1, default=str))
