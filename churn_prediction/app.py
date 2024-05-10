%%writefile app.py
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import torch
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained(
        "results"
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        'results'
    )
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model_and_tokenizer()

# Page title and description
st.title("Customer Churn Prediction")
st.write("Enter the customer details to predict if they will churn or not.")

# Form for user input
with st.form(key="input_form"):
    st.header("Customer Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        senior_citizen = st.radio("Senior Citizen", ["Yes", "No"])
        partner = st.radio("Partner", ["Yes", "No"])
        internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])
    with col2:
        dependents = st.radio("Dependents", ["Yes", "No"])
        multiple_lines = st.radio("Multiple Lines", ["Yes", "No"])
        contract = st.radio("Contract", ["Month-to-month", "One year", "Two year"])
        streaming_tv = st.radio("Streaming TV", ["Yes", "No"])
    with col3:
        streaming_movies = st.radio("Streaming Movies", ["Yes", "No"])
        paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
        tech_support = st.radio("Tech Support", ["Yes", "No"])
    tenure_months = st.number_input("Tenure Months", min_value=0)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
    total_charges = st.number_input("Total Charges", min_value=0.0, format="%.2f")
    submit_button = st.form_submit_button(label="Predict")

# Prediction logic
if submit_button:
    features = [
        f"Gender: {gender}, Senior: {senior_citizen}",
        f"Partner: {partner}, Dependents: {dependents}",
        (
            "has multiple lines"
            if multiple_lines == "Yes"
            else "does not have multiple lines"
        ),
        (
            f"uses {internet_service} internet service"
            if internet_service != "No"
            else "does not use internet service"
        ),
        f"is on a {contract} contract",
        (
            "subscribes to streaming TV"
            if streaming_tv == "Yes"
            else "does not subscribe to streaming TV"
        ),
        (
            "subscribes to streaming movies"
            if streaming_movies == "Yes"
            else "does not subscribe to streaming movies"
        ),
        (
            "uses paperless billing"
            if paperless_billing == "Yes"
            else "does not use paperless billing"
        ),
        "has tech support" if tech_support == "Yes" else "no tech support",
        f"Tenure: {tenure_months} months, Monthly charges: {monthly_charges} dollars, Total charges: {total_charges} dollars.",
    ]
    input_text = " ".join(features)
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = np.argmax(logits.numpy())
    result = "Churn" if predicted_class_id == 1 else "Not Churn"
    st.success(f"Prediction: **{result}**")
