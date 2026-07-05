import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

DATA_PATH = "Pile_data.txt"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep="\t")
    return df


FEATURES = ["Pile Diameter (mm)", "Pile Length (m)", "Ram Weight (kN)", "Drop Height (m)"]
TARGET_CANDIDATES = ["Pile Bearing Capacity (kN)", "Pile Bearing\nCapacity (kN)"]


def main():
    df = load_data()
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError("Target column not found in data.")
    cols = FEATURES + [target]
    for c in cols:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).replace("", np.nan).astype(float)
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    X = df[FEATURES].values
    y = df[target].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

    param_grid = {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [4, 6, 8]}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    base = xgb.XGBRegressor(objective="reg:squarederror", verbosity=0, random_state=42)
    grid = GridSearchCV(base, param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    def inv(a):
        return scaler_y.inverse_transform(np.array(a).reshape(-1, 1)).flatten()

    y_pred_test = inv(model.predict(X_test))
    y_true_test = inv(y_test)
    r2 = r2_score(y_true_test, y_pred_test)
    mae = mean_absolute_error(y_true_test, y_pred_test)
    rmse = mean_squared_error(y_true_test, y_pred_test) ** 0.5
    print(f"Test R2: {r2:.3f} | MAE: {mae:.2f} kN | RMSE: {rmse:.2f} kN")

    joblib.dump(model, "xgb_model.pkl")
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    print("Saved: xgb_model.pkl, scaler_X.pkl, scaler_y.pkl")


if __name__ == "__main__":
    main()
