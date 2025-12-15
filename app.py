import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("temperature_icecream_lr.pkl")

st.set_page_config(
    page_title="Temperature vs Ice Cream Sales",
    layout="centered"
)

st.title("🍦 Ice Cream Sales Prediction")
st.write("Predict ice cream sales based on temperature using Linear Regression.")

# User input
temperature = st.number_input(
    "Enter Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=30.0,
    step=0.5
)

# Prediction
if st.button("Predict Sales"):
    input_data = np.array([[temperature]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Ice Cream Sales: **{prediction[0]:.2f} units**")

st.markdown("---")
st.caption("Model: Linear Regression | Dataset: Temperature vs Ice Cream Sales")
