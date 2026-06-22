
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Load the original dataset to fit encoders and get imputation means
try:
    original_data = pd.read_csv('LoanApprovalPrediction.csv')
except FileNotFoundError:
    st.error("LoanApprovalPrediction.csv not found. Please ensure it's in the same directory as the app.")
    st.stop()

# Preprocessing steps (replicate from notebook)

# Drop Loan_ID for consistency with training
if 'Loan_ID' in original_data.columns:
    original_data = original_data.drop(['Loan_ID'], axis=1)

# Identify categorical and numerical columns for preprocessing
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
numerical_cols_with_nans = ['Dependents', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Fit LabelEncoders for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Handle potential NaN in original data before fitting if any
    # For robust production, ensure unique values are captured well
    le.fit(original_data[col].astype(str).unique())
    label_encoders[col] = le

# Calculate means for numerical imputation from the original data
imputation_means = original_data[numerical_cols_with_nans].mean().to_dict()


st.title('Loan Approval Prediction App')
st.write('Enter the applicant details to predict loan approval status.')

# User Inputs
st.header('Applicant Details')

gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents_options = ['0', '1', '2', '3+']
dependents = st.selectbox('Dependents', dependents_options)
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0.0, value=0.0)
loan_amount = st.number_input('Loan Amount (in thousands)', min_value=0.0, value=120.0)
loan_amount_term = st.selectbox('Loan Amount Term (in months)', [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0])
credit_history = st.selectbox('Credit History (1=yes, 0=no)', [1.0, 0.0])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Convert inputs to a DataFrame for prediction
if st.button('Predict Loan Status'):
    # Create a dictionary of inputs
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents, # Keep as string for now, convert after encoding if needed
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    input_df = pd.DataFrame([input_data])

    # Preprocessing for input_df
    for col in categorical_cols:
        # Ensure the value is in the encoder's known classes, or handle unknown
        # For simplicity, we assume user picks from valid options.
        # Convert to string to avoid issues with potential mixed types in selectbox output.
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

    # Handle 'Dependents' specifically for conversion if it was treated as numeric after encoding in original notebook
    # In the original notebook, Dependents was float64, not encoded by LabelEncoder. So, convert '3+' to '3.0'.
    input_df['Dependents'] = input_df['Dependents'].replace({'3+': '3.0'}).astype(float)
    
    # Impute missing values (though with Streamlit inputs, we expect complete data, 
    # this is for consistency with training preprocessing if we had NaNs in inputs)
    for col in numerical_cols_with_nans:
        if col in input_df.columns and input_df[col].isnull().any():
            input_df[col] = input_df[col].fillna(imputation_means[col])
            
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1: # Assuming 1 for 'Y' (Approved) and 0 for 'N' (Not Approved) based on label encoding
        st.success(f'Loan Status: Approved (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.error(f'Loan Status: Not Approved (Probability: {prediction_proba[0][0]:.2f})')

    st.write("Note: 'Approved' corresponds to 'Y' and 'Not Approved' to 'N' from the original dataset encoding.")
