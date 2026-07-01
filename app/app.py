import streamlit as st
import joblib
import numpy as np

# Load trained model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_prediction_model.pkl")

model = joblib.load(MODEL_PATH)

# App Title
st.title("🌱 Crop Recommendation System")

st.write("Enter the soil and weather details to predict the most suitable crop.")

# Input: Nitrogen
N = st.number_input("Nitrogen (N)", min_value=0, value=0)

# Input: Phosphorus
P = st.number_input("Phosphorus (P)", min_value=0, value=0)

# Input: Potassium
K = st.number_input("Potassium (K)", min_value=0, value=0)

# Input: Temperature
temperature = st.number_input("Temperature (°C)", value=25.0)

# Input: Humidity
humidity = st.number_input("Humidity (%)", value=50.0)

# Input: pH
ph = st.number_input("Soil pH", value=6.5)

# Input: Rainfall
rainfall = st.number_input("Rainfall (mm)", value=100.0)

# Predict Button
if st.button("Predict Crop"):

    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"🌾 Recommended Crop: {prediction[0]}")

