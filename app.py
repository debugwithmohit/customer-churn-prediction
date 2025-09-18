import streamlit as st
import joblib
import pandas as pd

# Load Model and Transformer
rf = joblib.load("rf_model_churn_compressed.pkl")
ct = joblib.load("ColumnTransformer_compressed.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to predict churn probability:")

# Basic Demographics
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

# Services
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Billing
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

# Numerical
tenure = st.number_input("Customer Tenure (in months)", min_value=0, max_value=72, value=12)
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0, max_value=200, value=70)
TotalCharges = st.number_input("Total Charges ($)", min_value=0, max_value=10000, value=500)

# ---- Create DataFrame ----
user_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

# ---- Prediction ----
if st.button("🔮 Predict Churn"):
    try:
        user_data_transformed = ct.transform(user_data)
        prob = rf.predict_proba(user_data_transformed)[0][1]

        st.write(f"Churn Probability: **{prob*100:.2f}%**")

        if prob > 0.5:
            st.error("⚠️ High Risk: Customer is likely to churn!")
        else:
            st.success("✅ Low Risk: Customer is likely to stay.")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
