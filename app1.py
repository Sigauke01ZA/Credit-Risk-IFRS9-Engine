import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load Lending Club Models
model = joblib.load(MODEL_DIR / "xgb_model.pkl")
model_columns = joblib.load(MODEL_DIR / "model_columns.pkl")

st.set_page_config(page_title="Lending Club PD Engine", layout="wide")
st.title("💳 Lending Club PD & ECL Engine")
st.markdown("Predict the Probability of Default (PD) and Expected Credit Loss (ECL) for an unsecured consumer loan.")

# ==========================================
# SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.header("⚙️ Borrower Profile")
    loan_amnt = st.number_input("Loan Amount (R)", min_value=1000, max_value=40000, value=15000)
    fico = st.slider("FICO Score", 600, 850, 700)
    annual_inc = st.number_input("Annual Income (R)", min_value=10000, max_value=500000, value=65000)
    dti = st.slider("Debt-to-Income (DTI) %", 0.0, 50.0, 20.0)
    term = st.selectbox("Term (Months)", [36, 60])
    grade = st.selectbox("Lending Club Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    emp_length = st.slider("Employment Length (Years)", 0, 10, 5)
    revol_util = st.slider("Revolving Utilization %", 0.0, 100.0, 40.0)
    delinq_2yrs = st.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
    inq_last_6mths = st.number_input("Inquiries (last 6 months)", 0, 10, 1)
    open_acc = st.number_input("Open Credit Lines", 0, 30, 8)
    total_acc = st.number_input("Total Credit Lines", 0, 80, 20)

# ==========================================
# PREDICTION LOGIC
# ==========================================
input_dict = {
    'loan_amnt': loan_amnt, 'term': term, 'emp_length': emp_length,
    'annual_inc': np.log1p(annual_inc), 'dti': dti, 'delinq_2yrs': delinq_2yrs,
    'fico_range_low': fico, 'inq_last_6mths': inq_last_6mths, 'open_acc': open_acc,
    'pub_rec': 0, 'revol_bal': 15000, 'revol_util': revol_util, 'total_acc': total_acc,
}
grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
input_dict['grade'] = grade_map[grade]

input_df = pd.DataFrame([input_dict])
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# Run Model
pd_prob = model.predict_proba(input_df)[0][1]
lgd = 0.9372
ecl = pd_prob * lgd * loan_amnt

# ==========================================
# OUTPUT ON MAIN SCREEN
# ==========================================
st.header("Model Output")

col1, col2, col3 = st.columns(3)
col1.metric("Probability of Default (PD)", f"{pd_prob:.2%}")
col2.metric("Assumed LGD", f"{lgd:.2%}")
col3.metric("Expected Credit Loss (ECL)", f"R {ecl:,.2f}")

st.info(f"**Interpretation:** This borrower has a {pd_prob:.2%} chance of defaulting. If they default, the bank expects to lose R {ecl:,.2f}.")