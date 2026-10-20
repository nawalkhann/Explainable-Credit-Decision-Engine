"""Explainable Credit Decision Engine - demo app.  Run: streamlit run app.py"""
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from config import ARTIFACTS, DATASETS, REASON_TEXT
from engine.letters import decision_notice

st.set_page_config(page_title="Explainable Credit Decision Engine", layout="wide")


@st.cache_resource
def load(name):
    with open(ARTIFACTS / name / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    metrics = json.load(open(ARTIFACTS / name / "metrics.json"))
    data = DATASETS[name]["loader"]()
    return bundle, metrics, data


name = st.sidebar.selectbox("Portfolio", list(DATASETS), format_func=lambda k: DATASETS[k]["title"])
cfg = DATASETS[name]
bundle, metrics, data = load(name)
sc, ml, cut = bundle["scorecard"], bundle["ml"], bundle["cutoffs"]
cutoff = st.sidebar.slider("Decline if ML probability of default is above", 0.05, 0.5,
                           float(cut["ml"]["cutoff_pd"]), 0.005)
st.sidebar.caption(f"Profit-optimal cut-off found in validation: {cut['ml']['cutoff_pd']:.1%}")

st.title("Explainable Credit Decision Engine")
st.caption("Champion scorecard (WoE/IV) and challenger ML model, with a reason for every decision.")

tab1, tab2, tab3, tab4 = st.tabs(["Score an applicant", "Batch scoring", "Model comparison", "Fairness and monitoring"])

with tab1:
    feats = cfg["features"]
    raw = data[feats]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Application")
        row = {}
        for f in feats:
            if f == "loan_to_income":
                continue
            s = raw[f]
            if pd.api.types.is_numeric_dtype(s):
                row[f] = st.number_input(f, value=float(s.median()), key=f)
            else:
                opts = sorted(s.dropna().astype(str).unique())
                mode = str(s.mode()[0])
                row[f] = st.selectbox(f, opts, index=opts.index(mode), key=f)
        x = pd.DataFrame([row])
        if cfg.get("recompute"):
            x = cfg["recompute"](x)
    with c2:
        p = float(ml.predict_proba(x)[0])
        p_sc = float(sc.predict_proba(x)[0])
        score = float(sc.score(x).iloc[0])
        ok = p < cutoff
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision", "APPROVE" if ok else "DECLINE")
        m2.metric("ML probability of default", f"{p:.1%}")
        m3.metric("Scorecard points", f"{score:.0f}", f"PD {p_sc:.1%}", delta_color="off")

        sv = ml.shap_values(x).iloc[0].sort_values()
        st.subheader("Why: contribution of each factor (SHAP, log-odds)")
        st.bar_chart(pd.DataFrame({"impact on risk": sv.values},
                                  index=[REASON_TEXT.get(i, i) for i in sv.index]), horizontal=True)

        reasons = [f for f, _ in ml.reason_codes(x)[0]]
        path = None if ok else ml.path_to_approval(x, cutoff, cfg["actionable"], cfg.get("recompute"))
        st.subheader("Customer notice")
        st.code(decision_notice("NEW", ok, p, score, reasons, path), language=None)

        with st.expander("Scorecard points breakdown (traditional view)"):
            pb = sc.points_breakdown(x).T
            pb.columns = ["points"]
            st.dataframe(pb)

with tab2:
    st.write("Upload a CSV with the same columns as the training data. Every row gets a "
             "decision, probability of default, scorecard points and four reason codes.")
    up = st.file_uploader("Applications CSV", type="csv")
    df_in = pd.read_csv(up) if up else data[cfg["features"]].sample(200, random_state=7)
    if not up:
        st.caption("No file uploaded: showing 200 sample applications from the portfolio.")
    if cfg.get("recompute"):
        df_in = cfg["recompute"](df_in)
    res = df_in.copy()
    res["pd_ml"] = ml.predict_proba(df_in).round(4)
    res["scorecard_points"] = sc.score(df_in).values
    res["decision"] = np.where(res["pd_ml"] < cutoff, "APPROVE", "DECLINE")
    rc = ml.reason_codes(df_in)
    for k in range(4):
        res[f"reason_{k + 1}"] = [REASON_TEXT.get(r[k][0], r[k][0]) if len(r) > k else "" for r in rc]
    st.dataframe(res, width="stretch")
    st.download_button("Download decisions", res.to_csv(index=False), "decisions.csv")

with tab3:
    a, b = metrics["scorecard"], metrics["ml"]
    st.table(pd.DataFrame({
        "Scorecard (WoE/IV)": [a["auc"], a["gini"], a["ks"], a["business"]["approval_rate"],
                               a["business"]["bad_rate_approved"], a["business"]["expected_profit_test"]],
        "ML (LightGBM + SHAP)": [b["auc"], b["gini"], b["ks"], b["business"]["approval_rate"],
                                 b["business"]["bad_rate_approved"], b["business"]["expected_profit_test"]]},
        index=["AUC", "Gini", "KS", "Approval rate", "Bad rate among approved", "Expected profit (test)"]))
    st.caption(f"The two models make the same decision on {metrics['agreement_rate']:.0%} of test applications.")
    cols = st.columns(3)
    for col, img in zip(cols * 2, ["roc", "profit", "calibration", "shap_importance", "iv"]):
        col.image(str(ARTIFACTS / name / f"{img}.png"))
    st.subheader("Scorecard points table")
    st.dataframe(sc.table, width="stretch")

with tab4:
    st.metric("Population stability index (train vs test scores)", metrics["psi_ml_score_train_vs_test"],
              help="Below 0.1 stable, 0.1-0.25 watch, above 0.25 investigate")
    if "fairness" in metrics:
        st.subheader("Fairness audit (protected attributes are not model inputs)")
        st.dataframe(pd.DataFrame(metrics["fairness"]), width="stretch")
    else:
        st.info("This portfolio has no protected attributes. Switch to the German credit portfolio "
                "for the fairness audit, or connect HMDA data.")
