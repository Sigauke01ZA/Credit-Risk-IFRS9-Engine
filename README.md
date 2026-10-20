# Credit-Risk-IFRS9-Engine
End-to-end Lending Club PD &amp; ECL model with Streamlit deployment.

# IFRS 9 Credit Risk & Expected Credit Loss (ECL) Engine

## Project Overview
This repository contains an end-to-end quantitative credit risk pipeline built on the Lending Club dataset (2007–2018). The objective was to build a Probability of Default (PD) model, calculate Expected Credit Loss (ECL) under IFRS 9, and conduct macroeconomic stress testing.

### Key Features
- **Data Engineering:** Extracted a 150,000-loan stratified sample from 1.34M loans, preserving the exact historical bad rate (20.15%). Built a documented data cleaning audit trail.
- **EDA & Vintage Analysis:** Visualized the correlation between origination volume spikes (2015) and subsequent default rate increases. 
- **PD Modeling:** 
  - *Champion Model:* Logistic Regression (Gini: 0.4058).
  - *Challenger Model:* XGBoost (Gini: 0.4153).
- **Model Calibration:** Applied Platt Scaling to align predicted PDs with actual default rates, correcting over-provisioning in the ECL engine.
- **LGD Estimation:** Calculated historical unsecured LGD at 93.72%.
- **Stress Testing:** Built a 3-tier macroeconomic sensitivity engine (0.5x to 2.5x PD shocks) demonstrating non-linear ECL impacts during recessions.
- **Deployment:** Built an interactive Streamlit app for real-time borrower scoring.

## Interactive Web App
To run the Streamlit dashboard locally:
1. Ensure `xgboost`, `streamlit`, `pandas`, and `joblib` are installed.
2. Run: `streamlit run app.py`

## Author
**Raymond Sibanda** | [LinkedIn](https://www.linkedin.com/in/raymond-sibanda)
FRM | BSc Mathematics
