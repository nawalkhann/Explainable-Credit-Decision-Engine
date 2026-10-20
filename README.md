# Explainable Credit Decision Engine

A working loan-decisioning engine that does what a bank credit team needs and what regulators ask for: every approve/decline comes with a probability of default, scorecard points, the top reasons in plain language, and, for declines, what the applicant could change to be approved.

It runs a **champion/challenger** setup side by side:

| | Champion: traditional scorecard | Challenger: ML model |
|---|---|---|
| Method | WoE binning, IV screening, logistic regression, points (600 at 50:1, PDO 20) | LightGBM with monotonic constraints |
| Explanations | Points lost per characteristic (classic adverse-action method) | SHAP contributions per applicant |
| Why it matters | What validators and regulators already accept | Better risk ranking and profit |

## Results on real data (held-out test set)

**Lending Club, 163,987 real loans (2007-2011), 18.3% default rate. Test set: 32,798 loans.**

| Metric | Scorecard | ML (LightGBM) |
|---|---|---|
| AUC | 0.669 | 0.688 |
| Gini | 0.338 | 0.375 |
| KS | 0.252 | 0.275 |
| Approval rate at profit-optimal cut-off | 49.4% | 49.1% |
| Default rate among approved | 11.0% | 10.2% |
| Expected profit on test loans | 6.46M | **7.86M (+22%)** |
| Approving everyone | -14.29M | -14.29M |

The two models agree on 87% of decisions. The ML model is well calibrated: in every risk decile predicted and actual default rates are within 1.4 points (`artifacts/lending_club/calibration_ml.csv`). Score stability between train and test: PSI 0.0002.

Profit uses stated assumptions (a good loan earns 12% of the amount, a bad loan loses 65%). Change them in `engine/governance.py` to match a bank's own pricing and loss-given-default.

**German credit, 1,000 real loans**: used for the fairness audit, because it has sex, age and foreign-worker fields. These are *never* model inputs; they are only used to check outcomes. Result: sex and foreign-worker status pass the four-fifths rule; **applicants under 25 do not (adverse impact ratio 0.68)**, which the report flags for review. Note: the test set is only 200 loans, so this is a demonstration of the check, not a conclusion.

## What makes it different

1. **Path to approval.** A declined applicant gets told not only *why* but *what would change it*, for example "reduce the loan from 20,500 to 14,800 and bring credit-card utilisation below 24%". Only fields a person can actually change are searched.
2. **Two explanation methods that cross-check each other.** If SHAP reasons and scorecard reasons disagree on an applicant, that case is worth a manual review.
3. **Leakage and proxy controls built in.** Lending Club's own interest rate is excluded (it is the lender's existing risk price, so using it just copies their model). State of residence is excluded as a possible proxy for protected groups.
4. **Governance in the box.** Calibration by decile, PSI drift monitoring, fairness audit, IV strength labels, full points table: the items a model-risk or validation team asks for.
5. **Profit-based cut-off.** The decline threshold is picked to maximise expected profit on validation data, then tested unchanged.

## Run it

```bash
pip install -r requirements.txt
bash data/download.sh     # fetches the public datasets
python train.py            # trains both portfolios, writes artifacts/
streamlit run app.py       # demo app
```

The app has four tabs: score one applicant (with SHAP chart, customer notice and path to approval), batch-score a CSV and download decisions with reason codes, model comparison, and fairness/monitoring.

## Project layout

```
config.py              datasets, features, reason-code wording
engine/scorecard.py    WoE/IV binning and points scorecard
engine/ml.py           LightGBM, SHAP reason codes, path to approval
engine/governance.py   AUC/Gini/KS, calibration, profit cut-off, fairness, PSI
engine/letters.py      decision notices
train.py               training, validation, charts, example notices
app.py                 Streamlit demo
artifacts/<dataset>/   metrics.json, models.pkl, charts, points table, audits
```

## Adding a bigger data source

Add an entry in `config.py` with a loader that returns a DataFrame with a `target` column (1 = default). Planned next sources:

- **Home Credit Default Risk** (Kaggle): application table plus bureau and previous-application tables; needs a Kaggle account.
- **Freddie Mac Single-Family Loan-Level**: mortgage origination plus monthly performance; needs free registration. Allows a true out-of-time test.
- **HMDA**: mortgage applications with race, sex and ethnicity, for a much stronger fairness audit.
- A bank's own extract, with the same loader pattern.

## Known limits (be upfront with buyers)

- The Lending Club extract has no application date, so the split is random rather than out-of-time. Freddie Mac data would fix this.
- Scorecard bins are quantile-based; a production scorecard would also enforce monotonic WoE and have a credit officer review the bins.
- Profit figures depend on the stated margin and loss assumptions.
- Reason wording should be reviewed by compliance before customer use.

## Rebuild the sales report

```bash
python reports/template/build_report.py . reports/credit_engine_report.html reports/template/report_tpl.html
```

## Report page

`docs/index.html` is the published report as a standalone page. Open it in a browser, or turn on GitHub Pages (Settings → Pages → branch `main`, folder `/docs`) to get a public link.
