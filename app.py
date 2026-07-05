import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

st.set_page_config(page_title="Pile Bearing Capacity Predictor", page_icon="🗎️", layout="wide", initial_sidebar_state="expanded")

FEATURES = ["Pile Diameter (mm)", "Pile Length (m)", "Ram Weight (kN)", "Drop Height (m)"]
TARGET = "Pile Bearing Capacity (kN)"
FEATURE_COLS = ["diameter", "length", "ram_weight", "drop_height"]

TRAIN_METRICS = {"R2": 0.9771, "R": 0.9885, "MAPE": 10.83, "RMSE": 131.56, "MAE": 94.94}

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

XGB_PARAMS = dict(n_estimators=300, learning_rate=0.01, max_depth=8, objective="reg:squarederror", verbosity=0, random_state=42)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(io.StringIO(DATA_CSV))
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_and_train():
    df = load_data()
    X = df[FEATURE_COLS].values.astype(float)
    y = df["pbc"].values.astype(float).reshape(-1, 1)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y).ravel()
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(Xs, ys)
    return model, x_scaler, y_scaler, df


def predict_pbc(model, x_scaler, y_scaler, values):
    arr = np.array(values, dtype=float).reshape(1, -1)
    arr_s = x_scaler.transform(arr)
    pred_s = model.predict(arr_s)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()[0]
    return float(pred)


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #0ea5e9 100%);
           padding: 2.2rem 2rem; border-radius: 16px; color: #fff; margin-bottom: 1.5rem;
           box-shadow: 0 8px 24px rgba(30,58,138,0.25);}
    .hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 0.3rem 0; font-weight: 700;}
    .hero p {color: #dbeafe; font-size: 1.02rem; margin: 0;}
    .metric-card {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px;
                  padding: 1.1rem 1.2rem; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                  border-top: 4px solid #2563eb; height: 100%;}
    .metric-card .label {color: #6b7280; font-size: 0.82rem; text-transform: uppercase;
                         letter-spacing: 0.04em; font-weight: 600;}
    .metric-card .value {color: #111827; font-size: 1.85rem; font-weight: 700; margin-top: 0.25rem;}
    .pred-box {background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: #fff;
               padding: 1.4rem 1.6rem; border-radius: 14px; margin-top: 0.5rem;
               box-shadow: 0 6px 18px rgba(5,150,105,0.3);}
    .pred-box .plabel {font-size: 0.9rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.04em;}
    .pred-box .pvalue {font-size: 2.4rem; font-weight: 800; line-height: 1.1;}
    section[data-testid="stSidebar"] {background: #f8fafc;}
    div.stButton > button {background: linear-gradient(135deg, #1e3a8a, #2563eb); color: #fff;
                           border: none; border-radius: 10px; padding: 0.55rem 1rem; font-weight: 600;
                           width: 100%;}
    div.stButton > button:hover {background: linear-gradient(135deg, #1e40af, #3b82f6); color: #fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

model, x_scaler, y_scaler, df = load_and_train()

st.markdown(
    """
    <div class="hero">
        <h1>Pile Bearing Capacity Predictor</h1>
        <p>Machine learning (XGBoost) estimation of axial pile bearing capacity from driving parameters.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("#### Model Performance (Training)")
m = TRAIN_METRICS
cards = [
    ("R\u00b2", f"{m['R2']:.4f}"),
    ("R", f"{m['R']:.4f}"),
    ("MAPE (%)", f"{m['MAPE']:.2f}"),
    ("RMSE (kN)", f"{m['RMSE']:.2f}"),
    ("MAE (kN)", f"{m['MAE']:.2f}"),
]
cols = st.columns(5)
for col, (label, value) in zip(cols, cards):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

st.sidebar.markdown("### \u2699\ufe0f Input Parameters")
st.sidebar.caption("Enter the pile driving parameters below.")
diameter = st.sidebar.number_input(FEATURES[0], min_value=50.0, max_value=1000.0, value=282.0, step=1.0)
length = st.sidebar.number_input(FEATURES[1], min_value=1.0, max_value=60.0, value=12.0, step=0.1)
ram_weight = st.sidebar.number_input(FEATURES[2], min_value=1.0, max_value=100.0, value=13.0, step=0.1)
drop_height = st.sidebar.number_input(FEATURES[3], min_value=0.1, max_value=6.0, value=1.0, step=0.1)
predict_clicked = st.sidebar.button("Predict Capacity", type="primary")

left, right = st.columns([1, 1])
with left:
    st.markdown("#### Prediction")
    if predict_clicked:
        pred = predict_pbc(model, x_scaler, y_scaler, [diameter, length, ram_weight, drop_height])
        st.markdown(
            f'<div class="pred-box"><div class="plabel">Predicted {TARGET}</div>'
            f'<div class="pvalue">{pred:,.1f} kN</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Set parameters in the sidebar and click **Predict Capacity** to estimate the pile bearing capacity.")
with right:
    st.markdown("#### Current Inputs")
    inp_df = pd.DataFrame({"Parameter": FEATURES, "Value": [diameter, length, ram_weight, drop_height]})
    st.dataframe(inp_df, hide_index=True, width="stretch")

st.divider()
with st.expander("\U0001f4c4 View training data ({} records)".format(len(df))):
    st.dataframe(df, width="stretch")

st.caption("Model: XGBoost Regressor | Features: pile diameter, length, ram weight, drop height.")
