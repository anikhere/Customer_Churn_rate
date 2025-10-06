# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model/customer_churn_ann.h5")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📞 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to leave the telecom company.")

# --- Input form ---
st.header("Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges ($)", 10.0, 120.0, 60.0)
total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Convert to dataframe for model
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "gender_Female": [1 if gender == "Female" else 0],
    "Partner_Yes": [1 if partner == "Yes" else 0],
    "Dependents_Yes": [1 if dependents == "Yes" else 0],
    "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
    "InternetService_No": [1 if internet_service == "No" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
    "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
    "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
    "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0],
})

# Scale numeric inputs
X_scaled = scaler.transform(input_data)

if st.button("🔍 Predict Churn"):
    prob = model.predict(X_scaled)[0][0]
    pred = int(prob > 0.5)
    st.subheader(f"📊 Churn Probability: {prob:.2f}")
    if pred == 1:
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is likely to STAY.")

st.caption("Built with ❤️ using TensorFlow + Streamlit")
