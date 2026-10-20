import json, html, sys
import pandas as pd
root = sys.argv[1]; outp = sys.argv[2]
m = json.load(open(f"{root}/artifacts/lending_club/metrics.json"))
g = json.load(open(f"{root}/artifacts/german/metrics.json"))
cal = pd.read_csv(f"{root}/artifacts/lending_club/calibration_ml.csv")
fa = pd.read_csv(f"{root}/artifacts/german/fairness_audit.csv")
sys.path.insert(0, root)
from config import REASON_TEXT
S, M = m["scorecard"], m["ml"]

def bars(items, cls, fmt):
    mx = max(v for _, v in items)
    return "".join(f'<div class="bar-row"><span class="bar-label">{html.escape(k)}</span>'
                   f'<span class="bar-track"><span class="bar {cls}" style="width:{v/mx*100:.1f}%"></span></span>'
                   f'<span class="bar-val">{fmt(v)}</span></div>' for k, v in items)

shap_items = [(REASON_TEXT.get(k, k).replace(" is high","").replace("High ",""), v) for k, v in list(m["shap_importance"].items())[:8]]
label = {"loan_to_income":"Loan to income","term":"Loan term","revol_util":"Credit card utilisation","dti":"Debt-to-income","annual_inc":"Annual income","verification_status":"Income verified","loan_amnt":"Loan amount","purpose":"Loan purpose","home_ownership":"Housing","total_acc":"Credit accounts","emp_length":"Years employed","longest_credit_length":"Credit history length","delinq_2yrs":"Missed payments, 2 yrs"}
shap_items = [(label[k], v) for k, v in list(m["shap_importance"].items())[:8]]
iv_items = [(label[r["feature"]], r["iv"]) for r in m["iv"][:8]]

# calibration svg
W, H, pl, pb, pt = 560, 240, 44, 34, 12
mx = 0.45
bw = (W - pl - 10) / 10
def y(v): return pt + (H - pt - pb) * (1 - v / mx)
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Predicted versus actual default rate by risk decile">']
for t in (0, .1, .2, .3, .4):
    svg.append(f'<line x1="{pl}" x2="{W-10}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
               f'<text x="{pl-6}" y="{y(t)+4:.1f}" class="ax" text-anchor="end">{int(t*100)}%</text>')
for i, r in cal.iterrows():
    x0 = pl + i * bw + bw * 0.15
    w = bw * 0.33
    svg.append(f'<rect x="{x0:.1f}" y="{y(r.predicted_pd):.1f}" width="{w:.1f}" height="{y(0)-y(r.predicted_pd):.1f}" class="b-ml"><title>Decile {r.band}: predicted {r.predicted_pd:.1%}</title></rect>')
    svg.append(f'<rect x="{x0+w+2:.1f}" y="{y(r.actual_default_rate):.1f}" width="{w:.1f}" height="{y(0)-y(r.actual_default_rate):.1f}" class="b-act"><title>Decile {r.band}: actual {r.actual_default_rate:.1%}</title></rect>')
    svg.append(f'<text x="{x0+w:.1f}" y="{H-pb+16}" class="ax" text-anchor="middle">{int(r.band)}</text>')
svg.append(f'<text x="{(W+pl)/2}" y="{H-2}" class="ax" text-anchor="middle">Risk decile (1 = safest)</text></svg>')
svg = "".join(svg)

fa_rows = "".join(
    f'<tr><td>{r.attribute.replace("_"," ")}</td><td>{r.group.replace("_"," ")}</td><td class="num">{r.n}</td>'
    f'<td class="num">{r.approval_rate:.1%}</td><td class="num">{r.adverse_impact_ratio:.2f}</td>'
    f'<td><span class="pill {"ok" if r.passes_4_5ths_rule else "flag"}">{"Pass" if r.passes_4_5ths_rule else "Review"}</span></td></tr>'
    for r in fa.itertuples())

notice = html.escape(m["example_notices"][0]["notice"])
mm = lambda v: f"{v/1e6:+.2f}M".replace("+","")
tpl = open(sys.argv[3]).read()
out = (tpl.replace("{{SHAP}}", bars(shap_items, "ml", lambda v: f"{v:.2f}"))
          .replace("{{IV}}", bars(iv_items, "sc", lambda v: f"{v:.3f}"))
          .replace("{{CAL}}", svg).replace("{{FAIR}}", fa_rows).replace("{{NOTICE}}", notice))
for k, v in {"S_AUC": f'{S["auc"]:.3f}', "M_AUC": f'{M["auc"]:.3f}', "S_GINI": f'{S["gini"]:.3f}', "M_GINI": f'{M["gini"]:.3f}',
             "S_KS": f'{S["ks"]:.3f}', "M_KS": f'{M["ks"]:.3f}',
             "S_APP": f'{S["business"]["approval_rate"]:.1%}', "M_APP": f'{M["business"]["approval_rate"]:.1%}',
             "S_BAD": f'{S["business"]["bad_rate_approved"]:.1%}', "M_BAD": f'{M["business"]["bad_rate_approved"]:.1%}',
             "S_PROF": mm(S["business"]["expected_profit_test"]), "M_PROF": mm(M["business"]["expected_profit_test"]),
             "ALL_PROF": mm(M["business"]["profit_if_approve_all"]),
             "UPLIFT": f'{M["business"]["expected_profit_test"]/S["business"]["expected_profit_test"]-1:.0%}',
             "AGREE": f'{m["agreement_rate"]:.0%}', "PSI": f'{m["psi_ml_score_train_vs_test"]:.4f}'}.items():
    out = out.replace("{{" + k + "}}", v)
assert "{{" not in out, out[out.index("{{"):out.index("{{")+30]
open(outp, "w").write(out)
print("ok", len(out))
