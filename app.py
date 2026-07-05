import numpy as np
import joblib
import streamlit as st

st.set_page_config(
    page_title="Pile Bearing Capacity Predictor",
    page_icon="\U0001F3D7\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["Pile Diameter (mm)", "Pile Length (m)", "Ram Weight (kN)", "Drop Height (m)"]

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); padding: 2rem 2.5rem; border-radius: 16px; color: #ffffff; margin-bottom: 1.5rem;}
.hero h1 {color: #ffffff; margin: 0; font-size: 2rem;}
.hero p {color: #dbeafe; margin: 0.4rem 0 0 0; font-size: 1rem;}
.result-card {background: #ecfdf5; border: 1px solid #10b981; border-radius: 14px; padding: 1.5rem; text-align: center; margin-top: 1rem;}
.result-value {font-size: 2.6rem; font-weight: 700; color: #047857; margin: 0;}
.result-label {font-size: 1rem; color: #065f46; margin: 0;}
.stButton>button {width: 100%; border-radius: 10px; height: 3rem; font-weight: 600;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

with st.sidebar:
    st.header("About")
    st.write("This tool predicts pile bearing capacity using a tuned XGBoost regression model.")
    st.markdown("**Inputs**")
    st.markdown("- Pile Diameter (mm)\n- Pile Length (m)\n- Ram Weight (kN)\n- Drop Height (m)")
    st.markdown("**Output**")
    st.markdown("- Pile Bearing Capacity (kN)")
    st.caption("Model: XGBoost | Features scaled with StandardScaler")

st.markdown(
    '<div class="hero"><h1>Pile Bearing Capacity Predictor</h1>'
    '<p>Estimate driven-pile bearing capacity (kN) from geometry and driving energy using an XGBoost model.</p></div>',
    unsafe_allow_html=True,
)

st.subheader("Input Parameters")
col1, col2 = st.columns(2)
with col1:
    diameter = st.number_input("Pile Diameter (mm)", min_value=0.0, value=300.0, step=10.0, help="Outer diameter of the pile in millimeters.")
    length = st.number_input("Pile Length (m)", min_value=0.0, value=15.0, step=0.5, help="Embedded length of the pile in meters.")
with col2:
    ram_weight = st.number_input("Ram Weight (kN)", min_value=0.0, value=40.0, step=1.0, help="Weight of the hammer ram in kilonewtons.")
    drop_height = st.number_input("Drop Height (m)", min_value=0.0, value=1.0, step=0.1, help="Hammer drop height in meters.")

st.markdown("")
predict = st.button("Predict Bearing Capacity", type="primary")

if predict:
    try:
        model, scaler_X, scaler_y = load_artifacts()
        X = np.array([[diameter, length, ram_weight, drop_height]])
        X_scaled = scaler_X.transform(X)
        pred_scaled = model.predict(X_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        st.markdown(
            f'<div class="result-card"><p class="result-label">Predicted Pile Bearing Capacity</p>'
            f'<p class="result-value">{pred:,.1f} kN</p></div>',
            unsafe_allow_html=True,
        )
        with st.expander("View input summary"):
            st.table({"Parameter": FEATURES, "Value": [diameter, length, ram_weight, drop_height]})
    except FileNotFoundError:
        st.error("Model files not found. Run train_model.py first to generate xgb_model.pkl and the scalers.")
