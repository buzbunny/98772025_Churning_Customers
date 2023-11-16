import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pickle

best_model_path = "best_model1.h5"
best_model = keras.models.load_model(best_model_path)

scaler_path = "scaler.pkl"
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


st.title("Churn Prediction Web _App")

st.sidebar.header("User Input Features")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.radio("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.radio("Partner", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure", min_value=0, max_value=72, value=36)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.radio("Paperless Billing", ["No", "Yes"])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=118.75, value=59.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=8684.8, value=4342.4)
internet_service_fiber_optic = st.sidebar.radio("Internet Service (Fiber Optic)", ["No", "Yes"])
online_security_no = st.sidebar.radio("Online Security (No)", ["No", "Yes"])
tech_support_no = st.sidebar.radio("Tech Support (No)", ["No", "Yes"])
payment_method_electronic_check = st.sidebar.radio("Payment Method (Electronic Check)", ["No", "Yes"])

if st.sidebar.button("Predict"):
    gender = 1 if gender == "Female" else 0
    senior_citizen = 1 if senior_citizen == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract = contract_mapping[contract]
    paperless_billing = 1 if paperless_billing == "Yes" else 0
    internet_service_fiber_optic = 1 if internet_service_fiber_optic == "Yes" else 0
    online_security_no = 1 if online_security_no == "Yes" else 0
    tech_support_no = 1 if tech_support_no == "Yes" else 0
    payment_method_electronic_check = 1 if payment_method_electronic_check == "Yes" else 0

    user_input = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "tenure": [tenure],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "InternetService_Fiber optic": [internet_service_fiber_optic],
            "OnlineSecurity_No": [online_security_no],
            "TechSupport_No": [tech_support_no],
            "PaymentMethod_Electronic check": [payment_method_electronic_check],
        }
    )

    input_features = ["tenure", "MonthlyCharges", "SeniorCitizen", "TotalCharges"]
    user_input[input_features] = scaler.transform(user_input[input_features])
   

    prediction = best_model.predict(user_input)

    st.header("Churn Prediction")

    confidence_rate = round(float(prediction[0]), 2) * 100
    st.write(f"The model is {confidence_rate}% confident about the prediction.")

    if prediction[0] > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")
