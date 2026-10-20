"""Train, validate and document the champion scorecard and challenger ML model.

Usage: python train.py --dataset lending_club   (or german, or all)
"""
import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ARTIFACTS, DATASETS, REPORTS
from engine.governance import (best_cutoff, calibration_table, discrimination,
                               fairness_audit, profit_curve, psi)
from engine.letters import decision_notice
from engine.ml import MLModel
from engine.scorecard import Scorecard

INK, ML_C, SC_C = "#1f2937", "#2563eb", "#d97706"


def run(name):
    cfg = DATASETS[name]
    out = ARTIFACTS / name
    out.mkdir(parents=True, exist_ok=True)
    df = cfg["loader"]()
    X, y = df[cfg["features"]], df["target"].astype(int)
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.25, stratify=y[idx_tr], random_state=42)
    print(f"[{name}] rows={len(df):,} bad_rate={y.mean():.3f} train={len(idx_tr):,} "
          f"val={len(idx_va):,} test={len(idx_te):,}")

    sc = Scorecard().fit(X.loc[idx_tr], y[idx_tr])
    ml = MLModel(cfg.get("monotone")).fit(X.loc[idx_tr], y[idx_tr], X.loc[idx_va], y[idx_va])

    p_sc, p_ml = sc.predict_proba(X.loc[idx_te]), ml.predict_proba(X.loc[idx_te])
    yt = y[idx_te].values
    metrics = {"scorecard": discrimination(yt, p_sc), "ml": discrimination(yt, p_ml)}

    # Business cut-off chosen on validation data, then applied unchanged to test
    amt = df[cfg["amount_col"]]
    cut = {}
    for k, m in (("scorecard", sc), ("ml", ml)):
        cv = profit_curve(y[idx_va].values, m.predict_proba(X.loc[idx_va]), amt[idx_va].values)
        c = best_cutoff(cv)["cutoff_pd"]
        p_t = p_sc if k == "scorecard" else p_ml
        test_curve = profit_curve(yt, p_t, amt[idx_te].values)
        app = p_t < c
        gain = np.where(yt[app] == 0, 0.12 * amt[idx_te].values[app], -0.65 * amt[idx_te].values[app])
        cut[k] = {"cutoff_pd": c, "approval_rate": round(float(app.mean()), 4),
                  "bad_rate_approved": round(float(yt[app].mean()), 4),
                  "expected_profit_test": round(float(gain.sum()), 0),
                  "profit_if_approve_all": round(float(test_curve["cum_profit"].iloc[-1]), 0)}
        metrics[k]["business"] = cut[k]

    metrics["psi_ml_score_train_vs_test"] = psi(ml.predict_proba(X.loc[idx_tr]), p_ml)
    metrics["iv"] = sc.binner.iv().round(4).reset_index(names="feature").to_dict("records")
    metrics["scorecard_features"] = sc.features
    metrics["dataset"] = {"name": name, "title": cfg["title"], "rows": len(df),
                          "bad_rate": round(float(y.mean()), 4)}

    # SHAP global importance on a test sample
    samp = X.loc[idx_te].sample(min(3000, len(idx_te)), random_state=0)
    sv = ml.shap_values(samp)
    imp = sv.abs().mean().sort_values(ascending=False)
    metrics["shap_importance"] = imp.round(4).to_dict()

    cal = calibration_table(yt, p_ml)
    cal.to_csv(out / "calibration_ml.csv")
    sc.table.to_csv(out / "scorecard_points_table.csv", index=False)

    # Fairness audit (protected attributes are not model inputs)
    if cfg["protected"]:
        fa = fairness_audit(df.loc[idx_te], p_ml, cut["ml"]["cutoff_pd"], yt, cfg["protected"])
        fa.to_csv(out / "fairness_audit.csv", index=False)
        metrics["fairness"] = fa.to_dict("records")

    # Example decisions with reasons from both models
    dec = pd.DataFrame({"pd_ml": p_ml, "pd_scorecard": p_sc, "score": sc.score(X.loc[idx_te]).values,
                        "actual_default": yt}, index=idx_te)
    dec["decision_ml"] = np.where(dec["pd_ml"] < cut["ml"]["cutoff_pd"], "APPROVE", "DECLINE")
    dec["decision_scorecard"] = np.where(dec["pd_scorecard"] < cut["scorecard"]["cutoff_pd"], "APPROVE", "DECLINE")
    metrics["agreement_rate"] = round(float((dec["decision_ml"] == dec["decision_scorecard"]).mean()), 4)
    dec.head(5000).to_csv(out / "test_decisions.csv")

    declined = dec[dec.decision_ml == "DECLINE"].sample(min(4, (dec.decision_ml == "DECLINE").sum()), random_state=1)
    approved = dec[dec.decision_ml == "APPROVE"].sample(2, random_state=1)
    notices = []
    for i, r in pd.concat([declined, approved]).iterrows():
        xr = X.loc[[i]]
        reasons = [f for f, _ in ml.reason_codes(xr)[0]]
        path = None
        if r.decision_ml == "DECLINE":
            path = ml.path_to_approval(df.loc[[i]], cut["ml"]["cutoff_pd"], cfg["actionable"], cfg.get("recompute"))
        notices.append({"id": int(i), "decision": r.decision_ml, "pd": float(r.pd_ml),
                        "score": float(r.score), "shap_reasons": reasons,
                        "scorecard_reasons": sc.reason_codes(xr)[0],
                        "notice": decision_notice(i, r.decision_ml == "APPROVE", r.pd_ml, r.score, reasons, path)})
    metrics["example_notices"] = notices

    charts(out, name, yt, p_sc, p_ml, imp, sc, cal, amt[idx_te].values)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(out / "models.pkl", "wb") as f:
        pickle.dump({"scorecard": sc, "ml": ml, "cutoffs": cut, "config_name": name}, f)
    print(json.dumps({k: metrics[k] for k in ("scorecard", "ml", "agreement_rate",
                                              "psi_ml_score_train_vs_test")}, indent=1))
    return metrics


def charts(out, name, yt, p_sc, p_ml, imp, sc, cal, amt):
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9ca3af", "font.size": 10})
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for p, lab, c in ((p_sc, "Scorecard (WoE)", SC_C), (p_ml, "ML (LightGBM)", ML_C)):
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], color="#d1d5db", lw=1, ls="--")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False)
    ax.set_title("ROC on held-out test loans", loc="left")
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    top = imp.head(10)[::-1]
    ax.barh(top.index, top.values, color=ML_C)
    ax.set_xlabel("Mean |SHAP| (log-odds)"); ax.set_title("What drives the ML model", loc="left")
    fig.tight_layout(); fig.savefig(out / "shap_importance.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    iv = sc.binner.iv()["iv"].head(10)[::-1]
    ax.barh(iv.index, iv.values, color=SC_C)
    ax.set_xlabel("Information Value"); ax.set_title("What drives the scorecard", loc="left")
    fig.tight_layout(); fig.savefig(out / "iv.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(cal["predicted_pd"], cal["actual_default_rate"], "o-", color=ML_C)
    m = max(cal.max())
    ax.plot([0, m], [0, m], color="#d1d5db", ls="--")
    ax.set_xlabel("Predicted default rate"); ax.set_ylabel("Actual default rate")
    ax.set_title("Calibration by risk decile (ML)", loc="left")
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for p, lab, c in ((p_sc, "Scorecard", SC_C), (p_ml, "ML", ML_C)):
        cv = profit_curve(yt, p, amt)
        ax.plot(cv["approval_rate"] * 100, cv["cum_profit"] / 1e6, color=c, lw=2, label=lab)
    ax.set_xlabel("Approval rate (%)"); ax.set_ylabel("Expected profit (millions)")
    ax.set_title("Profit vs approval rate", loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "profit.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    a = ap.parse_args()
    for n in (DATASETS if a.dataset == "all" else [a.dataset]):
        run(n)
