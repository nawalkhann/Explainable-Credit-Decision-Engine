"""Customer-facing decision notices built from reason codes."""
from config import REASON_TEXT

FIELD_LABEL = {"loan_amnt": "Loan amount", "amount": "Loan amount", "term": "Loan term",
               "duration": "Loan duration", "dti": "Debt-to-income ratio",
               "revol_util": "Credit card utilisation"}


def fmt(field, v):
    if field in ("loan_amnt", "amount", "annual_inc"):
        return f"{v:,.0f}"
    if field in ("term", "duration"):
        return f"{v:.0f} months"
    if field in ("dti", "revol_util"):
        return f"{v:.1f}%"
    return f"{v:.2f}"


def decision_notice(applicant_id, approved, pd_value, score, reasons, path=None):
    lines = [f"Application {applicant_id}",
             f"Decision: {'APPROVED' if approved else 'DECLINED'}",
             f"Estimated probability of default: {pd_value:.1%}   Scorecard points: {score:.0f}", ""]
    if approved:
        lines.append("Main factors that still added risk (for the credit file):")
    else:
        lines.append("We were unable to approve your application. The principal reasons were:")
    for i, f in enumerate(reasons, 1):
        lines.append(f"  {i}. {REASON_TEXT.get(f, f)}")
    if not approved:
        lines.append("")
        if path:
            lines.append("What would most likely change the outcome if you re-apply:")
            for f, old, new in path:
                lines.append(f"  - {FIELD_LABEL.get(f, f)}: from {fmt(f, old)} to {fmt(f, new)}")
        else:
            lines.append("No small change to the loan request alone would change this decision.")
        lines.append("")
        lines.append("You may request a manual review by a credit officer and a copy of the "
                     "information used in this decision.")
    return "\n".join(lines)
