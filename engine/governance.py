"""Model validation, business cut-off, fairness audit and drift monitoring."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve


def discrimination(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4),
            "ks": round(float(np.max(tpr - fpr)), 4), "brier": round(brier_score_loss(y, p), 4)}


def calibration_table(y, p, bins=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["band"] = pd.qcut(d["p"], bins, labels=False, duplicates="drop") + 1
    return d.groupby("band").agg(n=("y", "size"), predicted_pd=("p", "mean"),
                                 actual_default_rate=("y", "mean")).round(4)


def profit_curve(y, p, amount, margin=0.12, lgd=0.65):
    """Expected profit of approving everyone below each PD cut-off.
    Good loan earns margin * amount over its life; bad loan loses lgd * amount."""
    d = pd.DataFrame({"y": y, "p": p, "amt": amount}).sort_values("p")
    gain = np.where(d["y"] == 0, margin * d["amt"], -lgd * d["amt"])
    d["cum_profit"] = np.cumsum(gain)
    d["approval_rate"] = np.arange(1, len(d) + 1) / len(d)
    d["bad_rate_approved"] = np.cumsum(d["y"]) / np.arange(1, len(d) + 1)
    return d[["p", "approval_rate", "bad_rate_approved", "cum_profit"]].reset_index(drop=True)


def best_cutoff(curve):
    r = curve.loc[curve["cum_profit"].idxmax()]
    return {"cutoff_pd": round(float(r["p"]), 4), "approval_rate": round(float(r["approval_rate"]), 4),
            "bad_rate_approved": round(float(r["bad_rate_approved"]), 4),
            "expected_profit": round(float(r["cum_profit"]), 0)}


def fairness_audit(df, p, cutoff, y, protected):
    """Approval-rate parity (four-fifths rule) and equal opportunity
    (approval rate among applicants who actually repaid)."""
    approved = p < cutoff
    rows = []
    for col in protected:
        g = pd.DataFrame({"grp": df[col].values, "approved": approved, "good": np.asarray(y) == 0})
        rates = g.groupby("grp").apply(lambda s: pd.Series({
            "n": len(s), "approval_rate": s["approved"].mean(),
            "approval_rate_if_repaid": s.loc[s["good"], "approved"].mean()}), include_groups=False)
        ref = rates["approval_rate"].max()
        for grp, r in rates.iterrows():
            air = r["approval_rate"] / ref if ref else np.nan
            rows.append({"attribute": col, "group": grp, "n": int(r["n"]),
                         "approval_rate": round(r["approval_rate"], 3),
                         "approval_rate_if_repaid": round(r["approval_rate_if_repaid"], 3),
                         "adverse_impact_ratio": round(air, 3),
                         "passes_4_5ths_rule": bool(air >= 0.8)})
    return pd.DataFrame(rows)


def psi(expected, actual, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.25 watch, >0.25 investigate."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return round(float(np.sum((a - e) * np.log(a / e))), 4)
