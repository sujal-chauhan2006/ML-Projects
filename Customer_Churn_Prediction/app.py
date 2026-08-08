import streamlit as st
import pandas as pd
import joblib


# Load model
model = joblib.load(
    r"Joblib_File\customer_churn_pipeline.joblib"
)


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# Title
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer will churn.")


# Inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    value=12
)

usage_frequency = st.number_input(
    "Usage Frequency",
    min_value=0,
    value=10
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    value=2
)

payment_delay = st.number_input(
    "Payment Delay",
    min_value=0,
    value=5
)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Annual"]
)

total_spend = st.number_input(
    "Total Spend",
    min_value=0.0,
    value=1000.0
)

last_interaction = st.number_input(
    "Last Interaction",
    min_value=0,
    value=10
)


# Prediction button
if st.button("Predict Churn"):

    customer = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

    st.write(f"Churn Probability: {probability:.2%}")