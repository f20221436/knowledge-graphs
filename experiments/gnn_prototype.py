"""Prototype: message-passing GNN over each customer's temporal subgraph, inputs pulled ONLY from Neo4j.
Graph per customer: account node + 6 month nodes chained by NEXT_MONTH (edge features) + links to shared
RepaymentState nodes (learned embeddings). 5-fold CV on TRAIN customers only; test set untouched."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from credit_kg import graph as G, modeling as M, config as C

torch.manual_seed(C.SEED); np.random.seed(C.SEED); torch.set_num_threads(8)
STATES = ["NO_USE", "PAID_FULL", "REVOLVING", "LATE_1", "LATE_2PLUS"]
CHANGE = ["improving", "stable", "worsening"]

# ---- 1. pull the graph out of Neo4j --------------------------------------------------------------------
d = G.get_driver()
rec = G.run(d, """
MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord)-[:IN_STATE]->(s:RepaymentState)
OPTIONAL MATCH (m)-[r:NEXT_MONTH]->()
RETURN a.account_id AS cid, c.split AS split, a.credit_limit AS limit, m.month_index AS t, m.bill_amount AS bill,
       m.payment_amount AS pay, m.utilization AS util, m.is_late AS late, m.repayment_status AS status, s.code AS state,
       r.coverage AS cov, r.util_delta AS du, r.status_change AS chg""").sort_values(["cid", "t"])
cids = rec.cid.unique(); n = len(cids)
lim = rec.groupby("cid").limit.first().to_numpy()
R = lambda col: pd.to_numeric(rec[col], errors="coerce").to_numpy(float).reshape(n, 6)
bill, pay, util, status = R("bill"), R("pay"), R("util"), R("status")
node = np.stack([util.clip(0, 3), (bill / lim[:, None]).clip(-1, 3), (pay / lim[:, None]).clip(0, 3),
                 np.log1p(pay) / 12, np.log1p(bill.clip(0)) / 14, R("late"), status.clip(-2, 8) / 8,
                 (bill < 0).astype(float)], -1)                                     # (n, 6, 8)
state_idx = rec.state.map({s: i for i, s in enumerate(STATES)}).to_numpy().reshape(n, 6)
cov = R("cov")[:, :5]; du = R("du")[:, :5]
chg = rec["chg"].map({c: i for i, c in enumerate(CHANGE)}).fillna(1).to_numpy().reshape(n, 6)[:, :5].astype(int)
edge = np.concatenate([np.nan_to_num(cov, nan=0)[..., None], np.isnan(cov)[..., None].astype(float),
                       np.clip(du, -2, 2)[..., None], np.eye(3)[chg]], -1)           # (n, 5, 6)
acc = np.stack([np.log(lim) / 14], -1)                                                 # (n, 1)

labels = pd.read_parquet(C.PROCESSED / "customers.parquet")[["customer_id", C.TARGET]].set_index("customer_id")
y_all = labels.loc[cids, C.TARGET].to_numpy()
is_train = (rec.groupby("cid").split.first().to_numpy() == "train")
T = lambda a: torch.tensor(a, dtype=torch.float32)

# ---- 2. model --------------------------------------------------------------------------------------------
class ChainGNN(nn.Module):
    def __init__(self, h=48, layers=3, drop=0.1):
        super().__init__()
        self.state_emb = nn.Embedding(5, 8)                        # shared RepaymentState nodes
        self.inp = nn.Linear(8 + 8, h)
        self.acc = nn.Linear(1, h)
        self.fwd = nn.ModuleList(nn.Linear(h + 6, h) for _ in range(layers))   # message along NEXT_MONTH
        self.bwd = nn.ModuleList(nn.Linear(h + 6, h) for _ in range(layers))   # message against NEXT_MONTH
        self.slf = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.fromacc = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(4 * h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))

    def forward(self, x, s, e, a):
        h = torch.relu(self.inp(torch.cat([x, self.state_emb(s)], -1)))    # (b, 6, h)
        ha = torch.relu(self.acc(a))                                         # (b, h)
        for f, b_, sl, fa, nm in zip(self.fwd, self.bwd, self.slf, self.fromacc, self.norm):
            z = torch.zeros_like(h[:, :1])
            m_f = torch.cat([z, f(torch.cat([h[:, :-1], e], -1))], 1)       # month t-1 -> t
            m_b = torch.cat([b_(torch.cat([h[:, 1:], e], -1)), z], 1)       # month t+1 -> t
            h = nm(h + self.drop(torch.relu(sl(h) + m_f + m_b + fa(ha)[:, None])))
            ha = ha + h.mean(1)                                              # account node aggregates its months
        return self.head(torch.cat([h.mean(1), h.max(1).values, h[:, -1], ha], -1)).squeeze(-1)

def fit_predict(tr_idx, te_idx, epochs=60, bs=512):
    fit_idx, val_idx = train_test_split(tr_idx, test_size=0.1, stratify=y_all[tr_idx], random_state=C.SEED)
    X = [T(node), torch.tensor(state_idx), T(edge), T(acc)]
    y = T(y_all)
    model = ChainGNN(); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y_all.mean()) / y_all.mean()))
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(fit_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            lossf(model(*[t[b] for t in X]), y[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = roc_auc_score(y_all[val_idx], model(*[t[val_idx] for t in X]).numpy())
        if v > best: best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*[t[te_idx] for t in X])).numpy(), ep + 1

# ---- 3. same 5 folds as the RF checks; compare GNN, RF (A) and their average -------------------------
train_pos = np.where(is_train)[0]
table = M.load_table(); tr = table[table.split == "train"].set_index("customer_id").loc[cids[train_pos]]
params = json.load(open(C.MODELS / "rf_params.json"))
spec = json.loads((C.FEATURES / "feature_spec.json").read_text()); colsA = spec["raw_features"] + spec["engineered_features"]
cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
rows = []
for k, (a, b) in enumerate(cv.split(train_pos, y_all[train_pos])):
    t0 = time.time()
    p_gnn, eps = fit_predict(train_pos[a], train_pos[b])
    rf = M.make_rf(params["min_samples_leaf"]); rf.set_params(max_features=params["max_features"])
    p_rf = rf.fit(tr[colsA].iloc[a], y_all[train_pos[a]]).predict_proba(tr[colsA].iloc[b])[:, 1]
    yb = y_all[train_pos[b]]
    rank = lambda p: pd.Series(p).rank().to_numpy() / len(p)
    rows.append({"fold": k, "epochs": eps, "GNN (graph only)": roc_auc_score(yb, p_gnn), "RF A": roc_auc_score(yb, p_rf),
                 "RF A + GNN (avg rank)": roc_auc_score(yb, (rank(p_gnn) + rank(p_rf)) / 2),
                 "GNN PR-AUC": average_precision_score(yb, p_gnn), "RF PR-AUC": average_precision_score(yb, p_rf),
                 "secs": round(time.time() - t0)})
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
print(res.drop(columns=["fold"]).mean().round(4).to_string())
res.to_csv(C.TABLES / "gnn_prototype_cv.csv", index=False)

