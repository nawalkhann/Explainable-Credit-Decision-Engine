"""Dataset configurations. Adding a new data source (Home Credit, Freddie Mac, a bank's
own extract) means adding one entry here with a loader that returns a DataFrame with a
binary `target` column (1 = default / bad)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def load_lending_club():
    df = pd.read_csv(DATA / "lending_club_loans.csv")
    df = df.rename(columns={"bad_loan": "target"})
    df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
    # Debt burden derived feature: monthly instalment-free proxy of loan size vs income
    return add_lc_features(df)


def add_lc_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].clip(lower=1)
    return df


def load_german():
    df = pd.read_csv(DATA / "german_credit.csv")
    df["target"] = 1 - df["credit_risk"]  # source: 1 = good, 0 = bad
    df = df.drop(columns=["credit_risk"])
    df["sex"] = df["personal_status_sex"].str.split(" : ").str[0]
    df["age_group"] = pd.cut(df["age"], [0, 25, 200], labels=["under_25", "25_plus"]).astype(str)
    return df


DATASETS = {
    "lending_club": dict(
        loader=load_lending_club,
        title="Lending Club consumer loans (2007-2011, 163,987 real loans)",
        # int_rate is excluded on purpose: it is Lending Club's own risk price, so using it
        # would just copy the lender's existing model (target leakage for a new scorecard).
        # addr_state is excluded: geography can act as a proxy for protected groups.
        features=["loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
                  "purpose", "dti", "delinq_2yrs", "revol_util", "total_acc",
                  "longest_credit_length", "verification_status", "loan_to_income"],
        amount_col="loan_amnt",
        protected=[],
        # Fields an applicant can realistically change before re-applying
        actionable={"loan_amnt": "down", "term": "down", "dti": "down",
                    "revol_util": "down"},
        recompute=add_lc_features,
        monotone={"dti": 1, "revol_util": 1, "delinq_2yrs": 1, "annual_inc": -1,
                  "loan_to_income": 1, "term": 1, "longest_credit_length": -1},
    ),
    "german": dict(
        loader=load_german,
        title="German credit (1,000 real loans, used for the fairness audit)",
        features=["status", "duration", "credit_history", "purpose", "amount", "savings",
                  "employment_duration", "installment_rate", "other_debtors",
                  "present_residence", "property", "other_installment_plans", "housing",
                  "number_credits", "job", "people_liable", "telephone"],
        amount_col="amount",
        # Protected attributes are NOT model inputs; they are kept only to audit outcomes.
        protected=["sex", "age_group", "foreign_worker"],
        actionable={"amount": "down", "duration": "down"},
        recompute=None,
        monotone={"duration": 1, "amount": 1, "installment_rate": 1},
    ),
}

# Plain-language wording for reason codes (adverse action / decline letters)
REASON_TEXT = {
    "loan_amnt": "Requested loan amount is high",
    "term": "Longer repayment term",
    "emp_length": "Short employment history",
    "home_ownership": "Housing status",
    "annual_inc": "Income level",
    "purpose": "Purpose of the loan",
    "dti": "High debt-to-income ratio",
    "delinq_2yrs": "Recent missed payments (last 2 years)",
    "revol_util": "High use of existing credit lines",
    "total_acc": "Number of credit accounts",
    "longest_credit_length": "Length of credit history",
    "verification_status": "Income could not be verified",
    "loan_to_income": "Loan is large relative to income",
    "status": "Current account balance status",
    "duration": "Long loan duration",
    "credit_history": "Past repayment record",
    "amount": "Requested loan amount is high",
    "savings": "Low savings",
    "employment_duration": "Short time with current employer",
    "installment_rate": "Instalment is a high share of income",
    "other_debtors": "No guarantor or co-applicant",
    "present_residence": "Time at current address",
    "property": "Limited collateral or property",
    "other_installment_plans": "Other existing instalment plans",
    "housing": "Housing status",
    "number_credits": "Number of existing credits",
    "job": "Employment type",
    "people_liable": "Number of dependants",
    "telephone": "No registered telephone",
}
