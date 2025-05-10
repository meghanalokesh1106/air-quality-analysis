import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="PM2.5 Predictor", layout="centered")

# Load model and scaler with error handling
try:
    model = joblib.load("pm25_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("❌ Required model files not found (`pm25_prediction_model.pkl` and `scaler.pkl`).")
    st.stop()

# Custom style
st.markdown("""
    <style>
        .big-font { font-size: 24px !important; font-weight: bold; }
        .medium-font { font-size: 18px !important; }
        .result-box { 
            padding: 1.5rem;
            border-radius: 12px;
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            text-align: center;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='big-font'>🌍 PM2.5 Air Quality Predictor</div>", unsafe_allow_html=True)
st.markdown("Predict fine particulate pollution (PM2.5) using atmospheric readings. Enter the values below:")

# Input fields using number_input
with st.form(key="input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        pm10 = st.number_input("PM10 (μg/m³)", min_value=0, max_value=1000, value=100, step=10)
        temp = st.number_input("Temperature (°C)", min_value=-30, max_value=50, value=20, step=1)
        so2 = st.number_input("SO₂ (μg/m³)", min_value=0, max_value=500, value=50, step=5)

    with col2:
        no2 = st.number_input("NO₂ (μg/m³)", min_value=0, max_value=500, value=50, step=5)
        o3 = st.number_input("O₃ (μg/m³)", min_value=0, max_value=1000, value=150, step=10)
        pres = st.number_input("Pressure (hPa)", min_value=950, max_value=1050, value=1010, step=1)

    with col3:
        co = st.number_input("CO (μg/m³)", min_value=0, max_value=1000, value=300, step=10)

    submitted = st.form_submit_button("🔍 Predict")

# Prediction and output
if submitted:
    features = [[pm10, so2, no2, co, o3, temp, pres]]
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    def interpret_pm25(pm):
        if pm <= 12:
            return "Good 😊", "#4caf50"
        elif pm <= 35.4:
            return "Moderate 😐", "#cddc39"
        elif pm <= 55.4:
            return "Unhealthy for Sensitive Groups 😷", "#ff9800"
        elif pm <= 150.4:
            return "Unhealthy 😷", "#f44336"
        elif pm <= 250.4:
            return "Very Unhealthy 🤢", "#9c27b0"
        else:
            return "Hazardous ☠️", "#b71c1c"

    label, color = interpret_pm25(prediction)

    st.markdown(f"""
    <div class='result-box'>
        <div class='big-font' style='color:{color};'>Predicted PM2.5: {prediction:.2f} μg/m³</div>
        <div class='medium-font' style='color:{color};'>Air Quality: {label}</div>
    </div>
    """, unsafe_allow_html=True)
