"""Item 4: bank-style points scorecard (binning -> weight of evidence -> logistic regression -> points).

Bins are learned on TRAINING customers only. Adjacent bins are merged until the default rate moves in one
direction (monotone), except for features allowed a U-shape (utilization). Points use the standard
'points to double the odds' (PDO) scaling: score = offset + factor * ln(good:bad odds).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import load_config


def _woe_table(x: pd.Series, y: pd.Series, edges) -> pd.DataFrame:
    b = pd.cut(x, edges, include_lowest=True)
    t = pd.DataFrame({"bad": y.groupby(b, observed=False).sum(), "n": y.groupby(b, observed=False).size()})
    t["good"] = t["n"] - t["bad"]
    g, bd = t["good"].sum(), t["bad"].sum()
    t["woe"] = np.log(((t["good"] + 0.5) / g) / ((t["bad"] + 0.5) / bd))   # + = safer than average
    t["bad_rate"] = t["bad"] / t["n"].clip(lower=1)
    return t


def _bins(x: pd.Series, y: pd.Series, n_bins: int, min_share: float, monotone: bool) -> list:
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    edges = [-np.inf] + list(qs[1:-1]) + [np.inf]
    while len(edges) > 2:
        t = _woe_table(x, y, edges)
        small = (t["n"] / len(x)) < min_share
        rates = t["bad_rate"].to_numpy()
        d = np.diff(rates)
        nonmono = monotone and not ((d >= 0).all() or (d <= 0).all())
        if not small.any() and not nonmono:
            break
        # merge the pair of neighbouring bins with the most similar default rates
        i = int(np.argmin(np.abs(d)))
        edges.pop(i + 1)
    return edges


class Scorecard:
    def __init__(self, features, cfg=None):
        self.features = list(features)
        self.cfg = cfg or load_config()["scorecard"]

    def fit(self, X: pd.DataFrame, y) -> "Scorecard":
        y = pd.Series(np.asarray(y), index=X.index)
        c = self.cfg
        self.edges, self.tables = {}, {}
        for f in self.features:
            mono = f not in c["allow_non_monotone"]
            self.edges[f] = _bins(X[f], y, c["n_bins"], c["min_bin_share"], mono)
            self.tables[f] = _woe_table(X[f], y, self.edges[f])
        W = self._woe(X)
        self.lr = LogisticRegression(max_iter=2000).fit(W, y)   # models P(default)
        self.factor = c["pdo"] / np.log(2)
        self.offset = c["base_score"] - self.factor * np.log(c["base_odds"])
        return self

    def _woe(self, X):
        W = pd.DataFrame(index=X.index)
        for f in self.features:
            b = pd.cut(X[f], self.edges[f], include_lowest=True)
            W[f] = b.map(self.tables[f]["woe"]).astype(float).fillna(0.0)
        return W

    def points_table(self) -> pd.DataFrame:
        """Points per bin. ln(good:bad odds) = -(b0 + sum b_j * woe_j), so points = -b_j * woe * factor + share of offset."""
        b0, coefs = self.lr.intercept_[0], dict(zip(self.features, self.lr.coef_[0]))
        n = len(self.features)
        rows = []
        for f in self.features:
            t = self.tables[f]
            for interval, r in t.iterrows():
                pts = -coefs[f] * r["woe"] * self.factor + (self.offset - self.factor * b0) / n
                rows.append({"feature": f, "bin": str(interval), "customers_train": int(r["n"]),
                             "default_rate_train": round(r["bad_rate"], 3), "woe": round(r["woe"], 3), "points": round(pts)})
        return pd.DataFrame(rows)

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Total points (higher = safer)."""
        pt = self.points_table()
        total = pd.Series(0.0, index=X.index)
        for f in self.features:
            b = pd.cut(X[f], self.edges[f], include_lowest=True).astype(str)
            total += b.map(pt[pt.feature == f].set_index("bin")["points"]).astype(float).fillna(0)
        return total.rename("scorecard_points")

    def predict_proba(self, X) -> np.ndarray:
        return self.lr.predict_proba(self._woe(X))[:, 1]

    def explain(self, x_row: pd.DataFrame) -> pd.DataFrame:
        pt = self.points_table()
        rows = []
        for f in self.features:
            b = str(pd.cut(x_row[f], self.edges[f], include_lowest=True).iloc[0])
            p = pt[(pt.feature == f) & (pt.bin == b)]["points"].iloc[0]
            rows.append({"feature": f, "value": x_row[f].iloc[0], "bin": b, "points": p})
        return pd.DataFrame(rows)
