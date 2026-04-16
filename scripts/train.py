# scripts/train.py
# Purpose:
# - Load engineered features
# - Use only train rows
# - Time-based split
# - Compare multiple models
# - Track experiments in MLflow
# - Save best model and reports

import os
import time
import json
import joblib
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# Paths 
BASE = "."
DATA_PATH = os.path.join(BASE, "data", "features", "train_features.csv")
MODEL_DIR = os.path.join(BASE, "models")
OUTPUT_DIR = os.path.join(BASE, "reports")
MLRUNS_DIR = os.path.join(BASE, "mlruns")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)


# MLflow Setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("retail-demand-forecasting")


# Load Data
logging.info("Loading engineered dataset...")

df = pd.read_csv(DATA_PATH)

# Parse date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Keep only original train rows
if "is_train" in df.columns:
    df = df[df["is_train"] == 1].copy()

# Remove invalid essentials
df = df.dropna(subset=["date", "sales"]).reset_index(drop=True)

# Sort by time
df = df.sort_values("date").reset_index(drop=True)

logging.info(f"Rows available for training: {len(df)}")


# Columns to exclude
drop_cols = ["date"]

for col in ["id", "is_train"]:
    if col in df.columns:
        drop_cols.append(col)


# Time-based split
split_date = df["date"].quantile(0.8)

train_df = df[df["date"] <= split_date].copy()
val_df = df[df["date"] > split_date].copy()

X_train = train_df.drop(columns=["sales"] + drop_cols)
y_train = train_df["sales"]

X_val = val_df.drop(columns=["sales"] + drop_cols)
y_val = val_df["sales"]

logging.info(f"Split date: {split_date}")
logging.info(f"Maximum train date: {train_df['date'].max()}")
logging.info(f"Minimum validation date: {val_df['date'].min()}")
logging.info(f"Train rows: {len(X_train)}")
logging.info(f"Validation rows: {len(X_val)}")


# Fill remaining feature NaNs (lag/rolling)
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)


# Metrics
def evaluate(y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)

    denom = np.where(y_true == 0, 1, y_true)
    mape = np.mean(np.abs((y_true - pred) / denom)) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape)
    }


# Models
models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )
}


# Run naming
def get_run_name(name, model):

    if name == "LinearRegression":
        return "linreg_baseline_ts"

    if name == "RandomForest":
        p = model.get_params()
        return f"rf_est{p['n_estimators']}_d{p['max_depth']}_ts"

    if name == "XGBoost":
        p = model.get_params()
        lr = str(p["learning_rate"]).replace(".", "")
        return f"xgb_est{p['n_estimators']}_lr{lr}_d{p['max_depth']}_ts"

    return name


# Train Loop
results = []
best_model = None
best_name = None
best_rmse = float("inf")

for name, model in models.items():

    run_name = get_run_name(name, model)
    logging.info(f"Starting run: {run_name}")

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model_name", name)

        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)

        # Predict
        start_time = time.perf_counter()
        pred = model.predict(X_val)
        end_time = time.perf_counter()

        #latency
        latency_ms = (end_time - start_time) * 1000
        per_row_ms = latency_ms / len(X_val)
        mlflow.log_metric("prediction_latency_ms_batch", latency_ms)
        mlflow.log_metric("prediction_latency_ms_per_row", per_row_ms)
        logging.info(
            f"{run_name} latency: "
            f"{latency_ms:.2f} ms batch | "
            f"{per_row_ms:.4f} ms/row"
        )

        # Evaluate
        metrics = evaluate(y_val.values, pred)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Save model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        logging.info(f"{run_name} metrics: {metrics}")

        row = {"RunName": run_name, "Model": name}
        row.update(metrics)
        results.append(row)

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model = model
            best_name = run_name



# Register best model 
with mlflow.start_run(run_name=f"registry_{best_name}"):

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        registered_model_name="RetailDemandModel"
    )


# Save reports
with open(os.path.join(OUTPUT_DIR, "best_model_summary.json"), "w") as f:
    json.dump(
        {
            "best_run": best_name,
            "best_rmse": best_rmse
        },
        f,
        indent=4
    )

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)


# Feature Importance
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
else:
    importance = np.abs(best_model.coef_)

feat_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importance
}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feat_df["feature"], feat_df["importance"])
plt.gca().invert_yaxis()
plt.title(f"Feature Importance - {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200)
plt.close()


# Done
logging.info(f"Best run selected: {best_name}")
logging.info("Training completed successfully.")