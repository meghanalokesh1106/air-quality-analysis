import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
df = pd.read_csv('merged_data.csv')

# Remove rows with NaN values in important columns
df = df.dropna(subset=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES'])

# Prepare features and target
X = df[['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES']]
y = df['PM2.5']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'pm25_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler successfully saved!")