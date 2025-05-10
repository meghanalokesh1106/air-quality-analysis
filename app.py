import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Streamlit page configuration
st.set_page_config(page_title="China Air Quality Dashboard", layout="wide")

# Load model and scaler
try:
    model = joblib.load("pm25_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    model = None
    scaler = None

# Load dataset for EDA and overview
try:
    df = pd.read_csv("merged_data.csv")  # Replace with your actual file
except FileNotFoundError:
    df = None

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Navigation", ["Data Overview", "Exploratory Data Analysis (EDA)", "Modeling & Prediction"],
                           icons=["table", "bar-chart", "cpu"], menu_icon="cast", default_index=0)

# 1. Data Overview Page
if selected == "Data Overview":
    st.title("📊 Data Overview")
    if df is not None:
        # Station Selection
        st.write("### Filter Data by Station")
        station_options = df['station'].unique()
        selected_station = st.selectbox("Select Station", station_options)

        # Filter the dataset based on selected station
        filtered_data = df[df['station'] == selected_station]
        
        st.write(f"Showing data for {selected_station} station")

        # Show a sample of data
        st.write("### Sample of Data")
        st.dataframe(filtered_data.head())

        # Column Info
        st.write("### Column Info")
        st.write(filtered_data.dtypes)

        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(filtered_data.describe())

        # Missing Values
        st.write("### Missing Values")
        st.write(filtered_data.isnull().sum())
    else:
        st.warning("Dataset not found. Please ensure 'merged_data.csv' is available.")

# 2. EDA Page
elif selected == "Exploratory Data Analysis (EDA)":
    st.title("📈 Exploratory Data Analysis")
    if df is not None:
        # Station Selection for EDA
        st.write("### Filter Data by Station for EDA")
        selected_station_eda = st.selectbox("Select Station", df['station'].unique())
        filtered_data_eda = df[df['station'] == selected_station_eda]

        st.write(f"Exploring data for {selected_station_eda} station")

        # Display column names to help identify available columns
        st.write("### Column Names in Dataset")
        st.write(df.columns)

        # AQI Trends by Year
        st.subheader("AQI Trends")
        if 'date' in filtered_data_eda.columns:
            filtered_data_eda['date'] = pd.to_datetime(filtered_data_eda['date'])
            filtered_data_eda['year'] = filtered_data_eda['date'].dt.year
            year_aqi = filtered_data_eda.groupby('year')['PM2.5'].mean()
            st.line_chart(year_aqi)

        # Pollutant Correlations
        st.subheader("Pollutant Correlations")
        corr = filtered_data_eda.select_dtypes(include=[np.number]).corr()

        # Create a heatmap with adjustments to improve readability
        fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size for better spacing
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={'size': 10})  # Adjust annotation size
        st.pyplot(fig)

        # Additional EDA Visualizations
        st.subheader("Pollutant Distributions")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(filtered_data_eda['PM2.5'], kde=True, ax=ax, color='blue')
        ax.set_title('Distribution of PM2.5')
        st.pyplot(fig)

        # Show scatter plot options based on available columns
        st.write("### Scatter Plot of PM2.5 vs Other Pollutants")
        
        # Dynamically create a list of numeric columns for scatter plot
        numeric_columns = filtered_data_eda.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove 'PM2.5' from the list of numeric columns for scatter plot (as it will be used as y-axis)
        numeric_columns.remove('PM2.5')
        
        if numeric_columns:
            selected_pollutant = st.selectbox("Select pollutant for scatter plot:", numeric_columns)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=filtered_data_eda, x=selected_pollutant, y='PM2.5', ax=ax)
            ax.set_title(f'PM2.5 vs {selected_pollutant}')
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for scatter plot.")
            
    else:
        st.warning("Dataset not found. Please ensure 'merged_data.csv' is available.")

# 3. Modeling and Prediction Page
elif selected == "Modeling & Prediction":
    st.title("🤖 PM2.5 Air Quality Prediction")
    st.markdown("Predict fine particulate pollution (PM2.5) using atmospheric readings. Enter the values below:")

    if model is None or scaler is None:
        st.error("❌ Model or scaler not found. Please ensure `pm25_prediction_model.pkl` and `scaler.pkl` exist.")
    else:
        with st.form(key="prediction_form"):
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
            <div style='padding:1.5rem;border-radius:12px;background-color:#f0f2f6;border:1px solid #ddd;text-align:center;'>
                <div style='font-size:24px;font-weight:bold;color:{color};'>Predicted PM2.5: {prediction:.2f} μg/m³</div>
                <div style='font-size:18px;color:{color};'>Air Quality: {label}</div>
            </div>
            """, unsafe_allow_html=True)
