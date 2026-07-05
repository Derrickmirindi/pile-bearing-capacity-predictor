# Pile Bearing Capacity Predictor

An XGBoost regression model and an interactive Streamlit web application that predict the **pile bearing capacity (kN)** from four geotechnical inputs:

- Pile Diameter (mm)
- Pile Length (m)
- Ram Weight (kN)
- Drop Height (m)

## Project Structure

| File | Description |
| --- | --- |
| `train_model.py` | Loads and cleans the data, tunes XGBoost with 5-fold cross-validated grid search, evaluates on a held-out test set, and saves the model + scalers. |
| `app.py` | Streamlit app where users enter pile parameters and get a predicted bearing capacity. |
| `requirements.txt` | Python dependencies. |

## Data

Place your dataset as `Pile_data.txt` in the project root (tab-separated). It must contain the four feature columns above plus a target column named `Pile Bearing Capacity (kN)`.

## Setup

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

This produces `xgb_model.pkl`, `scaler_X.pkl`, and `scaler_y.pkl`, and prints the test R2, MAE, and RMSE.

## Run the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, enter the pile parameters, and click **Predict Bearing Capacity**.

## Deploy (Streamlit Community Cloud)

1. Push this repository (including the generated `.pkl` files) to GitHub.
2. Go to https://share.streamlit.io, connect your GitHub account, and select this repo.
3. Set the main file to `app.py` and deploy to get a shareable public URL.

## Model

Gradient-boosted decision trees (XGBoost) with hyperparameters selected via grid search over `n_estimators`, `learning_rate`, and `max_depth`. Features and target are standardized with `StandardScaler`.
