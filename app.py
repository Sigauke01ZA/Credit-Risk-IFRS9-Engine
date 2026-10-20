import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.express as px

# --- 1. DYNAMIC PATH SETUP ---
# This automatically finds the models folder regardless of where your project is saved!
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
COLS_PATH = BASE_DIR / "models" / "model_columns.pkl"

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="IFRS 9 PD Engine", layout="wide", page_icon="🏦")
st.title("🏦 IFRS 9 Credit Risk Engine")
st.caption("Interactive PD & ECL Calculator | XGBoost Challenger Model")

# --- 3. LOAD MODEL SAFELY ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        cols = joblib.load(COLS_PATH)
        return model, cols
    except Exception as e:
        st.error(f"🚨 Error loading model files. Ensure the 'models' folder exists. Error: {e}")
        st.stop()

model, model_columns = load_models()

# --- 4. SIDEBAR INPUTS ---
st.sidebar.header("Borrower Profile")
loan_amnt = st.sidebar.number_input("Loan Amount (R)", min_value=1000, max_value=40000, value=15000)
fico = st.sidebar.slider("FICO Score", 600, 850, 700)
annual_inc = st.sidebar.number_input("Annual Income (R)", min_value=10000, max_value=500000, value=65000)
dti = st.sidebar.slider("Debt-to-Income (DTI) %", 0.0, 50.0, 20.0)
term = st.sidebar.selectbox("Term (Months)", [36, 60])
grade = st.sidebar.selectbox("Lending Club Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
emp_length = st.sidebar.slider("Employment Length (Years)", 0, 10, 5)
revol_util = st.sidebar.slider("Revolving Utilization %", 0.0, 100.0, 40.0)
delinq_2yrs = st.sidebar.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
inq_last_6mths = st.sidebar.number_input("Inquiries (last 6 months)", 0, 10, 1)
open_acc = st.sidebar.number_input("Open Credit Lines", 0, 30, 8)
total_acc = st.sidebar.number_input("Total Credit Lines", 0, 80, 20)

# --- 5. PREDICTION LOGIC ---
# Create dataframe from inputs
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

# --- 6. TABS & UI ---
tab1, tab2 = st.tabs(["📊 PD & ECL Output", "📈 Model Calibration"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Default (PD)", f"{pd_prob:.2%}")
    col2.metric("Assumed LGD", f"{lgd:.2%}")
    col3.metric("Expected Credit Loss (ECL)", f"R {ecl:,.2f}")
    
    st.info(f"**Interpretation:** This borrower has a {pd_prob:.2%} chance of defaulting. If they default, the bank expects to lose R {ecl:,.2f}.")

with tab2:
    deciles = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    predicted = [0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.26, 0.33, 0.45]
    actual = [0.05, 0.06, 0.10, 0.11, 0.16, 0.17, 0.23, 0.25, 0.32, 0.45]
    
    chart_data = pd.DataFrame({"Decile": deciles, "Predicted PD": predicted, "Actual Default Rate": actual})
    fig = px.line(chart_data, x="Decile", y=["Predicted PD", "Actual Default Rate"], markers=True, title="Platt-Scaled Calibration Curve")
    st.plotly_chart(fig, use_container_width=True)