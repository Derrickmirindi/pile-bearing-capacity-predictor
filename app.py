import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

st.set_page_config(page_title="Pile Bearing Capacity Predictor", page_icon="PBC", layout="wide", initial_sidebar_state="expanded")

FEATURES = ["Pile Diameter (mm)", "Pile Length (m)", "Ram Weight (kN)", "Drop Height (m)"]
TARGET = "Pile Bearing Capacity (kN)"
FEATURE_COLS = ["diameter", "length", "ram_weight", "drop_height"]

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


@st.cache_resource(show_spinner=True)
def load_and_train():
    df = load_data()
    X = df[FEATURE_COLS].values.astype(float)
    y = df["pbc"].values.astype(float).reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y).ravel()

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2s, mapes, rmses, maes = [], [], [], []
    oof_true, oof_pred = [], []

    for train_idx, test_idx in kf.split(Xs):
        X_tr, X_te = Xs[train_idx], Xs[test_idx]
        y_tr, y_te = ys[train_idx], ys[test_idx]
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)
        pred_s = model.predict(X_te)
        pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
        true = y_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
        r2s.append(r2_score(true, pred))
        maes.append(mean_absolute_error(true, pred))
        rmses.append(np.sqrt(mean_squared_error(true, pred)))
        mask = true != 0
        mapes.append(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)
        oof_true.extend(true.tolist())
        oof_pred.extend(pred.tolist())

    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(Xs, ys)

    metrics = {
        "R2": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "MAPE": float(np.mean(mapes)),
        "RMSE": float(np.mean(rmses)),
        "MAE": float(np.mean(maes)),
    }
    return final_model, x_scaler, y_scaler, df, metrics, np.array(oof_true), np.array(oof_pred)


def predict_pbc(model, x_scaler, y_scaler, values):
    arr = np.array(values, dtype=float).reshape(1, -1)
    arr_s = x_scaler.transform(arr)
    pred_s = model.predict(arr_s)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()[0]
    return float(pred)


st.title("Pile Bearing Capacity Predictor")
st.caption("XGBoost model evaluated with 10-fold cross-validation")

model, x_scaler, y_scaler, df, metrics, oof_true, oof_pred = load_and_train()

c1, c2, c3, c4 = st.columns(4)
c1.metric("R2 (10-fold CV)", f"{metrics['R2']:.3f}")
c2.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
c3.metric("RMSE (kN)", f"{metrics['RMSE']:.1f}")
c4.metric("MAE (kN)", f"{metrics['MAE']:.1f}")

st.sidebar.header("Input Parameters")
diameter = st.sidebar.number_input(FEATURES[0], min_value=50.0, max_value=1000.0, value=282.0, step=1.0)
length = st.sidebar.number_input(FEATURES[1], min_value=1.0, max_value=60.0, value=12.0, step=0.1)
ram_weight = st.sidebar.number_input(FEATURES[2], min_value=1.0, max_value=100.0, value=13.0, step=0.1)
drop_height = st.sidebar.number_input(FEATURES[3], min_value=0.1, max_value=6.0, value=1.0, step=0.1)

if st.sidebar.button("Predict", type="primary"):
    pred = predict_pbc(model, x_scaler, y_scaler, [diameter, length, ram_weight, drop_height])
    st.success(f"Predicted {TARGET}: {pred:.1f} kN")

st.subheader("Cross-Validation: Actual vs Predicted")
fig = go.Figure()
fig.add_trace(go.Scatter(x=oof_true, y=oof_pred, mode="markers", name="Predictions", marker=dict(color="#1f77b4", size=7, opacity=0.7)))
lo = float(min(oof_true.min(), oof_pred.min()))
hi = float(max(oof_true.max(), oof_pred.max()))
fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal", line=dict(color="red", dash="dash")))
fig.update_layout(xaxis_title="Actual PBC (kN)", yaxis_title="Predicted PBC (kN)", height=500, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

with st.expander("View training data"):
    st.dataframe(df, use_container_width=True)
