import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

st.set_page_config(
page_title="Pile Bearing Capacity Predictor",
page_icon="🏷️",
layout="wide",
initial_sidebar_state="expanded",
)

FEATURES = ["Pile Diameter (mm)", "Pile Length (m)", "Ram Weight (kN)", "Drop Height (m)"]
TARGET = "Pile Bearing Capacity (kN)"

DATA_CSV = """diameter,length,ram_weight,drop_height,pbc
282,8,12,1,555
282,8,12,1,623
282,3,12,1,536
282,3,12,1,850
282,3,12,1,648
282,8,12,1,291
282,11,13,1.5,1572
282,11,13,1.5,1450
282,13,13,1.5,854
282,14,13,1.5,818
282,14,13,1.5,980
282,13,13,1.5,1063
395,28,35,1,1341
480,29,45,1,1409
480,29,45,1,2200
480,29,45,1,1650
226,10,13,1,1058
226,7,13,1,942
226,11,13,1,774
226,8,13,1,749
226,8,13,1,780
226,8,13,1,588
226,8,13,1,707
451,12,90,0.4,3530
306,17,90,0.3,2790
306,15,90,0.4,2900
451,23,90,0.4,3430
451,23,90,0.4,3460
226,17,25,0.4,780
226,17,25,0.4,770
169.63,3.4,12,0.5,410
199.2,13.31,23.1,1.34,1284.1
189.49,10.4,13,1,884
189.49,8.6,20,1,1480
189.49,7.6,18,1,870
169.63,9.7,13,1,789
169.63,15.5,13,1,910
211.1,10.3,13,1,817
211.1,9.3,13,1,970
189.49,4.6,13,1,829
189.49,5.2,13,1,410
189.49,8.1,13,1,625
189.49,7.6,13,1,567
189.49,15,13,1,754
169.63,15,13,1,1102
169.63,11.2,13,1,950
225.68,33.6,53,1.5,2750
252.31,18.8,50,1.5,2522
252.31,18.6,50,1.5,2459
169.63,11,13,1,792
169.63,10.9,13,1,819
189.49,6.65,13,1,500
189.49,6.9,13,1,700
189.49,8.6,20,1,1480
189.49,7.6,18,1,870
169.63,9.7,13,1,789
169.63,15.5,13,1,910
211.1,10.3,13,1,817
211.1,9.3,13,1,970
189.49,4.6,13,1,829
189.49,5.2,13,1,410
189.49,8.1,13,1,625
189.49,7.6,13,1,567
189.49,11.3,13,1,754
169.63,15,13,1,1102
169.63,11.2,13,1,950
225.68,33.6,53,1.5,2750
252.31,18.8,50,1.5,2522
252.31,18.6,50,1.5,2459
169.63,11,13,1,792
169.63,10.9,13,1,819
189.49,6.65,13,1,500
189.49,6.9,13,1,700
189.49,8,13,1,751
189.49,8,13,1,568
189.49,16.9,25,1,1141
239.63,32,63,4,3449
239.63,15.3,63,4,3227
239.63,28.3,63,4,3692
239.37,33.2,63,3,2680
189.49,19.7,15,1,1419
189.49,10.5,15,1,1150
189.49,11.2,15,1,898
169.63,11.4,13,1,774
169.63,7.9,13,1,707
189.49,9,20,0.5,1040
195.44,8.2,18,1,1200
189.49,9.5,15,0.7,980
189.49,8.1,12,1,623
189.49,3.4,12,1,648
189.49,8.6,20,1,1480
211.1,10,13,1,811
169.63,9.7,13,1,789
211.1,10.3,13,1,817
211.1,12.7,13,1,983
189.49,11.3,13,1,754
225.68,33.6,53,1.5,2750
189.49,6.9,13,1,505
189.49,8,13,1,568
239.63,28.3,63,4,3692
239.37,33.2,63,3,2680
169.63,11.4,13,1,774
189.49,10.5,15,0.7,1040
282,8,12,1,555
282,8,12,1,623
282,3,12,1,536
282,3,12,1,850
282,8,12,1,648
282,8,12,1,291
226,10,13,1,318
226,9,13,1,858
282,23,13,1,560
282,23,13,1,427
282,22,13,1,979
282,23,13,1,568
282,22,13,1,533
282,23,13,1,586
282,22,13,1,934
282,22,13,1,749
282,22,13,1,753
282,23,13,1,810
282,15,20,1,669
282,9,20,1,1480
282,10,20,1,1225
400,35,25,1,829
400,35,25,1,605
300,35,25,1,772
"""

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

@st.cache_resource(show_spinner="Training XGBoost model...")
def load_and_train():
    df = pd.read_csv(io.StringIO(DATA_CSV))
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    X = df[["diameter", "length", "ram_weight", "drop_height"]].values
    y = df["pbc"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.1, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=8,
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred_test = scaler_y.inverse_transform(
        model.predict(X_test).reshape(-1, 1)
    ).flatten()
    y_true_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = {
        "r2": r2_score(y_true_test, y_pred_test),
        "mape": mape(y_true_test, y_pred_test),
        "rmse": float(np.sqrt(mean_squared_error(y_true_test, y_pred_test))),
        "mae": mean_absolute_error(y_true_test, y_pred_test),
    }

    all_pred = scaler_y.inverse_transform(
        model.predict(X_scaled).reshape(-1, 1)
    ).flatten()

    return model, scaler_X, scaler_y, df, metrics, y, all_pred

model, scaler_X, scaler_y, df, metrics, y_all, all_pred = load_and_train()

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f4c81 0%, #2a9d8f 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.05rem; }
    .result-card {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-left: 6px solid #2a9d8f;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .result-value { font-size: 3rem; font-weight: 800; color: #0f4c81; }
    .result-label { font-size: 1rem; color: #555; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏷️ Pile Bearing Capacity Predictor</h1>
        <p>Estimate axial pile bearing capacity (kN) from driving parameters using an XGBoost model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Input Parameters")
diameter = st.sidebar.number_input(
    "Pile Diameter (mm)", min_value=50.0, max_value=1000.0, value=282.0, step=1.0
)
length = st.sidebar.number_input(
    "Pile Length (m)", min_value=1.0, max_value=60.0, value=12.0, step=0.1
)
ram_weight = st.sidebar.number_input(
    "Ram Weight (kN)", min_value=1.0, max_value=150.0, value=20.0, step=1.0
)
drop_height = st.sidebar.number_input(
    "Drop Height (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
)
predict = st.sidebar.button("Predict Capacity", type="primary", use_container_width=True)

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Prediction")
    if predict:
        x_new = np.array([[diameter, length, ram_weight, drop_height]])
        x_new_scaled = scaler_X.transform(x_new)
        pred_scaled = model.predict(x_new_scaled).reshape(-1, 1)
        pred = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Pile Bearing Capacity</div>
                <div class="result-value">{pred:,.0f} kN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Inputs \u2192 Diameter: {diameter} mm | Length: {length} m | "
            f"Ram: {ram_weight} kN | Drop: {drop_height} m"
        )
    else:
        st.info("Set the parameters in the sidebar and click **Predict Capacity**.")

with col2:
    st.subheader("XGBoost Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("R\u00b2", f"{metrics['r2']:.3f}")
    m2.metric("MAPE (%)", f"{metrics['mape']:.2f}")
    m3, m4 = st.columns(2)
    m3.metric("RMSE (kN)", f"{metrics['rmse']:.1f}")
    m4.metric("MAE (kN)", f"{metrics['mae']:.1f}")
    st.caption(
        f"Trained on {len(df)} records with XGBoost "
        f"(StandardScaler, n_estimators=300, lr=0.01, max_depth=8, 90/10 split)."
    )

st.divider()
st.subheader("Predicted vs Measured (all records)")
lims = [float(min(y_all.min(), all_pred.min())), float(max(y_all.max(), all_pred.max()))]
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=y_all,
        y=all_pred,
        mode="markers",
        name="Records",
        marker=dict(color="#2a9d8f", size=8, opacity=0.7),
    )
)
fig.add_trace(
    go.Scatter(
        x=lims,
        y=lims,
        mode="lines",
        name="Ideal (y = x)",
        line=dict(color="#0f4c81", dash="dash"),
    )
)
fig.update_layout(
    xaxis_title="Measured PBC (kN)",
    yaxis_title="Predicted PBC (kN)",
    height=460,
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View training data"):
    st.dataframe(df, use_container_width=True)
