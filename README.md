# CreditSight

**Alternative credit scoring for the unbanked — behavioral signals, not paperwork.**

🔗 **Live app:** [creditsightt.streamlit.app](https://creditsightt.streamlit.app/)

---

## The problem

Traditional credit scoring (CIBIL, FICO) relies entirely on formal credit history — past loans, credit cards, EMI records. Hundreds of millions of people never touched formal credit at all: gig workers, daily-wage earners, first-time earners, students. They're not necessarily risky borrowers — the system simply has no data trail to evaluate them on, so it defaults to rejecting them or pushing them toward predatory informal lenders.

## What CreditSight does

CreditSight predicts creditworthiness using **alternative behavioral signals** instead of formal credit history — payment regularity, spending discipline, and income stability. It's built for two audiences:

- **Individuals** checking where they stand, with a plain-language explanation of what's driving their score and what would improve it
- **Banks / NBFCs** reviewing a batch of applicants at once, with an adjustable approval threshold

Every score comes with a reason, not just a number — this is a hard requirement in real-world lending, since regulators require lenders to explain why a loan was denied.

## Features

**Individual Score**
- Score computed live from 11 behavioral inputs (credit utilization, payment history, income stability, etc.)
- SHAP-based signal breakdown — see exactly which factors pushed the score up or down, ranked by impact
- Indicative loan affordability estimate based on risk tier and existing obligations
- "What would improve your score?" simulator — drag key factors and see the score update live
- CIBIL-scale comparison and percentile context for intuition

**Portfolio Review (Bank / NBFC)**
- Upload a CSV of applicants and score them all at once
- Adjustable approval threshold with live approved/declined breakdown
- Risk-tier distribution (Low / Moderate / High)
- Export scored results as CSV

**Trust & transparency**
- "Why you can trust this score" section with the technical model validation available on request
- Regulatory context panel covering the RBI Digital Lending framework
- Dark/light theme, fully keyboard-accessible, WCAG-checked contrast

## How it works

1. **Data**: Kaggle's "Give Me Some Credit" dataset, reframed — columns like `RevolvingUtilizationOfUnsecuredLines` and late-payment counts were relabeled and reinterpreted as behavioral proxies (spending discipline, payment regularity) rather than raw credit-bureau fields.
2. **Feature engineering**: two composite signals were built — a weighted payment-regularity score (severe lateness weighted more than minor lateness) and an income-stability tier derived from debt-to-income ratio.
3. **Modeling**: three models were trained and honestly compared on the same held-out test set — Logistic Regression, Random Forest, and XGBoost — using precision, recall, and AUC (not accuracy, since only ~7% of the data is the default class).

   | Model | ROC-AUC | Precision (default) | Recall (default) |
   |---|---|---|---|
   | Logistic Regression | 0.826 | 0.24 | 0.65 |
   | Random Forest (0.5 threshold) | 0.841 | 0.57 | 0.16 |
   | Random Forest (0.3 threshold) | 0.841 | 0.44 | 0.36 |
   | **XGBoost (selected)** | 0.835 | 0.24 | 0.64 |

   XGBoost was selected over Random Forest despite a marginally lower AUC, because Random Forest's recall on actual defaulters collapsed to 0.16 at default settings — missing 84% of real risk despite looking more "accurate" overall. XGBoost's `scale_pos_weight` parameter held recall nearly level with the simpler baseline while still improving on it.
4. **Explainability**: SHAP (`TreeExplainer`) generates per-applicant feature attributions, converted into plain-English factors in the app.
5. **App**: built in Streamlit, with the trained model exported via XGBoost's native JSON format and the SHAP explainer rebuilt fresh from the loaded model at runtime.

## Tech stack

- **Modeling**: Python, pandas, scikit-learn, XGBoost, SHAP — developed in Google Colab
- **App**: Streamlit
- **Data source**: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)

## Running locally

```bash
pip install streamlit xgboost shap pandas scikit-learn numpy
streamlit run app.py
```

Requires `xgb_model.json` (trained model) in the same directory as `app.py`. A `.streamlit/config.toml` is included to set the app's base theme.

## Disclaimer

CreditSight is a demonstration of alternative credit scoring built on reframed, publicly available data. It is not a licensed credit bureau product, does not access real financial accounts, and its output should not inform actual lending or borrowing decisions.

## Credits

Built by **Oorvi Kulshreshtha**.
