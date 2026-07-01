import streamlit as st
import joblib
import numpy as np
import os
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_prediction_model.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "images", "farm.jpeg")

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)

# ---------------- BACKGROUND IMAGE ----------------
def add_bg(image_path):
    with open(image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main > div {{
            background: rgba(255,255,255,0.82);
            padding: 30px;
            border-radius: 20px;
            margin-top: 20px;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
        }}

        h1 {{
            text-align:center;
            color:#0B6623;
            font-size:40px;
        }}

        p {{
            text-align:center;
            font-size:18px;
        }}

        .stButton>button {{
            width:100%;
            background:linear-gradient(90deg,#28a745,#0B6623);
            color:white;
            font-size:18px;
            border:none;
            border-radius:12px;
            padding:12px;
            font-weight:bold;
        }}

        .stButton>button:hover {{
            background:linear-gradient(90deg,#0B6623,#28a745);
            color:white;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg(IMAGE_PATH)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 About")

st.sidebar.info(
    """
Crop Recommendation System

Developed using:
- Python
- Streamlit
- Scikit-Learn
- Random Forest
- Joblib

Developer:
Vivek Pathade
"""
)

# ---------------- TITLE ----------------
st.title("🌾 Crop Recommendation System")

st.write(
    "Enter the soil nutrients and weather conditions to get the most suitable crop recommendation."
)

# ---------------- INPUTS ----------------

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, value=90)
    P = st.number_input("Phosphorus (P)", min_value=0, value=42)
    K = st.number_input("Potassium (K)", min_value=0, value=43)
    temperature = st.number_input("Temperature (°C)", value=20.8)

with col2:
    humidity = st.number_input("Humidity (%)", value=82.0)
    ph = st.number_input("Soil pH", value=6.5)
    rainfall = st.number_input("Rainfall (mm)", value=202.9)

st.write("")

# ---------------- PREDICTION ----------------

if st.button("🌱 Predict Crop"):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = model.predict(input_data)

    st.success(f"🌾 Recommended Crop: **{prediction[0].upper()}**")

    st.balloons()

    st.markdown("---")

    st.markdown(
        f"""
        <div style="background:#d4edda;
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    box-shadow:0px 5px 20px rgba(0,0,0,0.2);">

        <h2 style="color:#155724;">
        🌾 {prediction[0].upper()}
        </h2>

        <h4 style="color:#155724;">
        This crop is recommended based on the given soil and weather conditions.
        </h4>

        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
<div style="text-align:center;color:white;font-size:16px;">
Developed by <b>Vivek Pathade</b> ❤️ using Streamlit
</div>
""",
    unsafe_allow_html=True,
)