# air-quality-analysis


 ## 🌍 China Air Quality Insights Dashboard
This Streamlit web application offers an interactive platform to predict and analyze PM2.5 pollution levels based on various environmental factors. The goal of this project is to predict fine particulate matter (PM2.5) using machine learning models, helping to assess air quality and provide actionable insights.

## 🔍 Project Overview
 The dataset includes hourly air quality measurements like PM2.5, PM10, SO₂, NO₂, CO, and O₃ from cities in China (2013-2017). It also contains weather data, including temperature, pressure, and more. The data was pre-processed and cleaned, handling missing values, outliers, and feature engineering to build the predictive model.

 ## ✅ Key Processing Steps
 Missing Values Handling: Numerical features were filled with the mean value, and categorical features with the mode.

Outlier Treatment: Winsorization was used to clip extreme values.

Feature Engineering:

Created a timestamp feature.

Estimated AQI (Air Quality Index) based on multiple pollutants.

Categorized pollution sources (e.g., vehicle, industrial).

Lag Features: Generated lag features to capture temporal patterns.

Feature Scaling & Encoding:

Applied LabelEncoder for categorical features and StandardScaler for numerical ones.

Feature Selection: Used Random Forest for feature importance analysis and selected the top 10 most influential features.

## 📊 Visualizations
Interactive Plots: Visualizations help explore:

AQI trends by year and station.

Monthly AQI variations.

Pollution patterns linked to vehicle and industrial sources.

## 🤖 Modeling & Prediction
The following machine learning models were trained to predict PM2.5:

Random Forest Regressor

AdaBoost Regressor

XGBoost Regressor

Data was split based on station and time to avoid data leakage, ensuring accurate time-series forecasting. Model evaluation used metrics like MAE, MSE, RMSE, and R² Score. A scatter plot visualizes predicted vs. actual AQI values.

## 📌 Features of the App
The PM2.5 Predictor Streamlit app allows users to input environmental variables and predict PM2.5 levels with a real-time display of results. The features include:

Predict PM2.5: Enter values such as PM10, temperature, SO₂, NO₂, CO, O₃, and pressure.

Visualization: Display the predicted PM2.5 levels and classify air quality based on the Air Quality Index (AQI).

Prediction Output: The app shows the predicted PM2.5 value and categorizes the air quality into various levels such as Good, Moderate, Unhealthy, etc.

Performance Metrics: The app provides key performance metrics from the trained models (e.g., MAE, RMSE).

User-Friendly Interface: Simple, interactive inputs, and quick predictions.

## Model File Download

You can download the `pm25_prediction_model.pkl` model from the following OneDrive link:

[Download pm25_prediction_model.pkl]  https://1drv.ms/u/c/1c8a8f0924ff7f74/Ed-febDxW9hGpwZRoxT8bzsB0lg8WU_SSWyBs_OO16S4FQ?e=knbpCz
