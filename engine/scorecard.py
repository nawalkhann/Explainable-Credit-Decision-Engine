"""Traditional bank scorecard: Weight of Evidence binning, Information Value screening,
logistic regression on WoE, and points scaling (PDO method)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 0.5  # smoothing so empty good/bad cells do not give infinite WoE


def _iv_label(iv):
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (check leakage)"


class WoEBinner:
    """Quantile bins for numeric fields, category bins (rare levels pooled) for text.
    Missing values get their own bin, which is how banks usually treat them."""

    def __init__(self, max_bins=8, min_share=0.05):
        self.max_bins, self.min_share = max_bins, min_share
        self.bins, self.tables = {}, {}

    def _bin_series(self, col, s):
        spec = self.bins[col]
        if spec["kind"] == "num":
            out = pd.cut(s, spec["edges"], include_lowest=True).astype(str)
        else:
            out = s.astype(str).where(s.astype(str).isin(spec["levels"]), "OTHER")
        return out.where(s.notna(), "MISSING")

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                vals = np.unique(s.dropna())
                if len(vals) <= self.max_bins:  # few distinct values: one bin each
                    cuts = (vals[:-1] + vals[1:]) / 2
                else:
                    cuts = np.unique(np.nanquantile(s.dropna(), np.linspace(0, 1, self.max_bins + 1)))[1:-1]
                edges = np.concatenate([[-np.inf], cuts, [np.inf]])
                self.bins[col] = {"kind": "num", "edges": edges}
            else:
                share = s.astype(str).value_counts(normalize=True)
                self.bins[col] = {"kind": "cat", "levels": list(share[share >= self.min_share / 2].index)}
            b = self._bin_series(col, s)
            t = pd.DataFrame({"bin": b, "y": y}).groupby("bin", observed=True)["y"].agg(["count", "sum"])
            t.columns = ["n", "bad"]
            t["good"] = t["n"] - t["bad"]
            gd = (t["good"] + EPS) / (t["good"].sum() + EPS)
            bd = (t["bad"] + EPS) / (t["bad"].sum() + EPS)
            t["bad_rate"] = t["bad"] / t["n"]
            t["woe"] = np.log(gd / bd)
            t["iv"] = (gd - bd) * t["woe"]
            self.tables[col] = t
        return self

    def iv(self):
        r = pd.Series({c: t["iv"].sum() for c, t in self.tables.items()}).sort_values(ascending=False)
        return pd.DataFrame({"iv": r, "strength": r.map(_iv_label)})

    def transform(self, X):
        out = {}
        for col in X.columns:
            if col not in self.tables:
                continue
            b = self._bin_series(col, X[col])
            out[col] = b.map(self.tables[col]["woe"]).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    """Points = offset - factor * ln(odds of bad). Default: 600 points at 50:1 good:bad,
    20 points to double the odds. Higher score = lower risk."""

    def __init__(self, base_score=600, base_odds=50, pdo=20, min_iv=0.02):
        self.base_score, self.base_odds, self.pdo, self.min_iv = base_score, base_odds, pdo, min_iv
        self.binner = WoEBinner()

    def fit(self, X, y):
        self.binner.fit(X, y)
        iv = self.binner.iv()
        self.features = [f for f in iv.index if iv.loc[f, "iv"] >= self.min_iv]
        W = self.binner.transform(X[self.features])
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit(W, y)
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        self._build_points()
        return self

    def _build_points(self):
        n = len(self.features)
        b0 = self.lr.intercept_[0]
        rows = []
        for f, beta in zip(self.features, self.lr.coef_[0]):
            t = self.binner.tables[f]
            for bin_label, woe in t["woe"].items():
                # model predicts bad, so ln(odds good) = -(b0 + sum beta*woe)
                pts = self.offset / n + self.factor * (b0 / n + beta * woe) * -1
                rows.append({"feature": f, "bin": bin_label, "n": int(t.loc[bin_label, "n"]),
                             "bad_rate": round(t.loc[bin_label, "bad_rate"], 4),
                             "woe": round(woe, 4), "points": round(pts, 1)})
        self.table = pd.DataFrame(rows)

    def predict_proba(self, X):
        return self.lr.predict_proba(self.binner.transform(X[self.features]))[:, 1]

    def points_breakdown(self, X):
        """Points per characteristic for each applicant (columns = features)."""
        out = {}
        for f in self.features:
            b = self.binner._bin_series(f, X[f])
            lookup = self.table[self.table.feature == f].set_index("bin")["points"]
            out[f] = b.map(lookup).fillna(lookup.median())
        return pd.DataFrame(out, index=X.index)

    def score(self, X):
        return self.points_breakdown(X).sum(axis=1).round(0)

    def reason_codes(self, X, top=4):
        """Classic adverse-action method: rank characteristics by points lost versus
        the best attainable points for that characteristic."""
        pb = self.points_breakdown(X)
        best = self.table.groupby("feature")["points"].max()
        lost = best[pb.columns].values - pb
        return [list(lost.loc[i].sort_values(ascending=False).head(top).index) for i in pb.index]
