import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model & scaler
with open('xgb_loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Loan Approval Prediction App")

# User Inputs
person_age = st.number_input("Person Age", min_value=18, max_value=100, value=25)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.number_input("Annual Income", min_value=1000, value=50000)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=2)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_amnt = st.number_input("Loan Amount", min_value=500, value=5000)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=0.1)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=3)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Previous Defaults", ["Yes", "No"])

# Preprocessing user input
# Encode education 
education_map = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4
}

# Encode previous defaults
previous_defaults_map = {"Yes": 1, "No": 0}

# All training columns in exact order
model_columns = [
    'person_age','person_education','person_income','person_emp_exp','loan_amnt',
    'loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score',
    'previous_loan_defaults_on_file','person_gender_male','loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT','loan_intent_MEDICAL','loan_intent_PERSONAL',
    'loan_intent_VENTURE','person_home_ownership_OTHER','person_home_ownership_OWN',
    'person_home_ownership_RENT'
]

# Create input dictionary with all zeros first
input_dict = {col: [0] for col in model_columns}

# Fill in actual values
input_dict['person_age'] = [person_age]
input_dict['person_gender_male'] = [1 if person_gender=='male' else 0]
input_dict['person_education'] = [education_map[person_education]]
input_dict['person_income'] = [person_income]
input_dict['person_emp_exp'] = [person_emp_exp]
input_dict[f'person_home_ownership_{person_home_ownership}'] = [1]
input_dict['loan_amnt'] = [loan_amnt]
input_dict[f'loan_intent_{loan_intent}'] = [1]
input_dict['loan_int_rate'] = [loan_int_rate]
input_dict['loan_percent_income'] = [loan_percent_income]
input_dict['cb_person_cred_hist_length'] = [cb_person_cred_hist_length]
input_dict['credit_score'] = [credit_score]
input_dict['previous_loan_defaults_on_file'] = [previous_defaults_map[previous_loan_defaults_on_file]]

# Convert to DataFrame in same column order
input_df = pd.DataFrame(input_dict)[model_columns]

# Scale numeric columns
num_cols = ['person_age','person_income','person_emp_exp','loan_amnt',
            'loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Display results
st.subheader("Prediction Result")
pred_prob = model.predict_proba(input_df)[:,1][0]
st.write(f"Approval Probability: **{pred_prob:.2f}**")
