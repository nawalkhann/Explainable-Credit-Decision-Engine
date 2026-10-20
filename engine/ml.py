"""Challenger ML model (LightGBM) with SHAP reason codes and counterfactual
'path to approval' suggestions."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import shap
warnings.filterwarnings("ignore", category=UserWarning, module="shap")


def prepare(X, categories=None):
    """Turn text columns into pandas categoricals with a fixed level set."""
    X = X.copy()
    categories = categories or {}
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            if c not in categories:
                categories[c] = sorted(X[c].dropna().astype(str).unique())
            X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return X, categories


class MLModel:
    def __init__(self, monotone=None):
        # monotone: {feature: +1 risk rises with value, -1 risk falls}. Monotonic
        # constraints make the model's behaviour defensible to a validator.
        self.monotone = monotone or {}

    def fit(self, X, y, X_val, y_val):
        X, self.categories = prepare(X)
        X_val, _ = prepare(X_val, self.categories)
        self.features = list(X.columns)
        mono = [self.monotone.get(f, 0) for f in self.features]
        self.model = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            monotone_constraints=mono, verbose=-1)
        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="auc",
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def predict_proba(self, X):
        X, _ = prepare(X[self.features], self.categories)
        return self.model.predict_proba(X)[:, 1]

    def shap_values(self, X):
        Xp, _ = prepare(X[self.features], self.categories)
        sv = self.explainer.shap_values(Xp)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return pd.DataFrame(sv, columns=self.features, index=X.index)

    def reason_codes(self, X, top=4):
        """Features pushing this applicant's risk up the most (positive SHAP, log-odds)."""
        sv = self.shap_values(X)
        out = []
        for i in sv.index:
            row = sv.loc[i]
            row = row[row > 0].sort_values(ascending=False).head(top)
            out.append([(f, float(v)) for f, v in row.items()])
        return out

    def path_to_approval(self, x_row, cutoff, actionable, recompute=None, max_steps=12):
        """Greedy search over actionable fields for the smallest change that brings PD
        under the cutoff. Returns list of (field, old, new) or None if not reachable."""
        x = x_row.copy()
        start = x_row.copy()
        if self.predict_proba(x)[0] < cutoff:
            return []
        for _ in range(max_steps):
            best, best_pd = None, self.predict_proba(x)[0]
            for f, direction in actionable.items():
                if f not in self.features or pd.isna(x[f].iloc[0]):
                    continue
                trial = x.copy()
                v = trial[f].iloc[0]
                if f == "term":
                    if v <= 36:
                        continue
                    trial[f] = 36.0
                else:
                    trial[f] = v * (0.85 if direction == "down" else 1.15)
                if recompute:
                    trial = recompute(trial)
                p = self.predict_proba(trial)[0]
                if p < best_pd - 1e-4:
                    best, best_pd = trial, p
            if best is None:
                return None
            x = best
            if best_pd < cutoff:
                return [(f, start[f].iloc[0], x[f].iloc[0]) for f in actionable
                        if f in x.columns and x[f].iloc[0] != start[f].iloc[0]]
        return None
