import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="Pile Bearing Capacity Predictor", page_icon="", layout="centered")

FEATURES = ["Pile Diameter (mm)", "Pile Length (m)", "Ram Weight (kN)", "Drop Height (m)"]

@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

st.title("Pile Bearing Capacity Predictor")
st.write("Enter the pile parameters below to estimate the bearing capacity (kN) using an XGBoost model.")

col1, col2 = st.columns(2)
with col1:
    diameter = st.number_input("Pile Diameter (mm)", min_value=0.0, value=300.0, step=10.0)
    length = st.number_input("Pile Length (m)", min_value=0.0, value=15.0, step=0.5)
with col2:
    ram_weight = st.number_input("Ram Weight (kN)", min_value=0.0, value=40.0, step=1.0)
    drop_height = st.number_input("Drop Height (m)", min_value=0.0, value=1.0, step=0.1)

if st.button("Predict Bearing Capacity", type="primary"):
    try:
        model, scaler_X, scaler_y = load_artifacts()
        X = np.array([[diameter, length, ram_weight, drop_height]])
        X_scaled = scaler_X.transform(X)
        pred_scaled = model.predict(X_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        st.success(f"Predicted Pile Bearing Capacity: {pred:,.1f} kN")
    except FileNotFoundError:
        st.error("Model files not found. Run train_model.py first to generate xgb_model.pkl and the scalers.")
