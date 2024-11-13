import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and input columns
Model = joblib.load("model.pkl")
Inputs = joblib.load("Inputs.pkl")

# Define the prediction function
def Predicitons(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, 
                CoapplicantIncome, LoanAmount, Credit_History, Property_Area, 
                loan_term_in_years):
    # Calculate Total_income and Income_to_Loan_Ratio
    Total_income = ApplicantIncome + CoapplicantIncome
    Income_to_Loan_Ratio = Total_income / LoanAmount if LoanAmount != 0 else 0

    # Prepare the input DataFrame
    pr_df = pd.DataFrame(columns=Inputs)
    pr_df.at[0, 'Gender'] = Gender
    pr_df.at[0, 'Married'] = Married
    pr_df.at[0, 'Dependents'] = Dependents
    pr_df.at[0, 'Education'] = Education
    pr_df.at[0, 'Self_Employed'] = Self_Employed
    pr_df.at[0, 'ApplicantIncome'] = ApplicantIncome
    pr_df.at[0, 'CoapplicantIncome'] = CoapplicantIncome
    pr_df.at[0, 'LoanAmount'] = LoanAmount
    pr_df.at[0, 'Credit_History'] = Credit_History
    pr_df.at[0, 'Property_Area'] = Property_Area
    pr_df.at[0, 'loan_term_in_years'] = loan_term_in_years
    pr_df.at[0, 'Total_income'] = Total_income
    pr_df.at[0, 'Income_to_Loan_Ratio'] = Income_to_Loan_Ratio
    
    # Predict loan status
    prediction = Model.predict(pr_df)
    return prediction[0]

# Streamlit main function
def main():
    st.title("Loan Prediction")

    # User inputs
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Married = st.selectbox("Married", ['Yes', 'No'])
    Dependents = st.slider("Dependents", min_value=0, max_value=3, value=2, step=1)
    Education = st.selectbox("Education", ['Not Graduate', 'Graduate'])
    Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
    ApplicantIncome = st.slider("Applicant Income", min_value=150, max_value=10000, value=2000, step=500)
    CoapplicantIncome = st.slider("Coapplicant Income", min_value=0, max_value=5701, value=1500, step=500)
    LoanAmount = st.slider("Loan Amount", min_value=30, max_value=244, value=100, step=10)
    Credit_History = st.selectbox("Credit History", [0, 1])
    Property_Area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])
    loan_term_in_years = st.slider("Loan Term in Years", min_value=1, max_value=30, value=5, step=1)

    # Calculate and display derived fields
    Total_income = ApplicantIncome + CoapplicantIncome
    Income_to_Loan_Ratio = Total_income / LoanAmount if LoanAmount != 0 else 0
    st.write(f"Total Income: {Total_income}")
    st.write(f"Income to Loan Ratio: {Income_to_Loan_Ratio:.2f}")

    # Prediction
    if st.button("Predict Loan Status"):
        result = Predicitons(Gender, Married, Dependents, Education, Self_Employed, 
                             ApplicantIncome, CoapplicantIncome, LoanAmount, 
                             Credit_History, Property_Area, loan_term_in_years)
        st.success(f"Predicted Loan Status: {result}")

if __name__ == "__main__":
    main()
