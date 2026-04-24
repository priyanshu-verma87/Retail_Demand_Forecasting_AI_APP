# scripts/train.py

import os
import time
import json
import yaml
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# Helpers
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


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


# Main training function
def run_training():

    params = load_params()

    # Paths
    DATA_PATH = params["data"]["train_path"]
    MODEL_DIR = params["paths"]["model_dir"]
    OUTPUT_DIR = params["paths"]["output_dir"]
    MLRUNS_DIR = params["paths"]["mlruns_dir"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Load Data
    logging.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "is_train" in df.columns:
        df = df[df["is_train"] == 1].copy()

    df = df.dropna(subset=["date", "sales"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    logging.info(f"Rows available: {len(df)}")

    # Drop columns
    drop_cols = ["date"]

    for col in ["id", "is_train"]:
        if col in df.columns:
            drop_cols.append(col)

    # Time split
    split_ratio = params["split"]["validation_ratio"]
    split_index = int(len(df) * (1 - split_ratio))

    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()

    X_train = train_df.drop(columns=["sales"] + drop_cols).fillna(0)
    y_train = train_df["sales"]

    X_val = val_df.drop(columns=["sales"] + drop_cols).fillna(0)
    y_val = val_df["sales"]

    logging.info(f"Train rows: {len(X_train)}")
    logging.info(f"Validation rows: {len(X_val)}")

    rs = params["general"]["random_state"]

    # Models
    models = {}

    if params["models"]["linear_regression"]:
        models["LinearRegression"] = LinearRegression()

    if params["models"]["random_forest"]["enabled"]:
        rf = params["models"]["random_forest"]

        models["RandomForest"] = RandomForestRegressor(
            n_estimators=rf["n_estimators"],
            max_depth=rf["max_depth"],
            n_jobs=rf["n_jobs"],
            random_state=rs
        )

    if params["models"]["xgboost"]["enabled"]:
        xgb = params["models"]["xgboost"]

        models["XGBoost"] = XGBRegressor(
            n_estimators=xgb["n_estimators"],
            learning_rate=xgb["learning_rate"],
            max_depth=xgb["max_depth"],
            subsample=xgb["subsample"],
            colsample_bytree=xgb["colsample_bytree"],
            objective="reg:squarederror",
            n_jobs=xgb["n_jobs"],
            random_state=rs
        )

    # Training Loop
    results = []
    best_model = None
    best_name = None
    best_rmse = float("inf")

    for name, model in models.items():

        run_name = get_run_name(name, model)

        logging.info(f"Running {run_name}")

        with mlflow.start_run(run_name=run_name):

            mlflow.log_param("model_name", name)

            if hasattr(model, "get_params"):
                for k, v in model.get_params().items():
                    if isinstance(v, (int, float, str, bool)):
                        mlflow.log_param(k, v)

            train_start = time.perf_counter()

            model.fit(X_train, y_train)

            pred = model.predict(X_val)

            train_end = time.perf_counter()

            metrics = evaluate(y_val.values, pred)
            logging.info(
                f"{run_name} | "
                f"MAE={metrics['MAE']:.4f}, "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}, "
                f"MAPE={metrics['MAPE']:.2f}%"
            )

            latency_ms = (train_end - train_start) * 1000
            per_row_ms = latency_ms / len(X_val)

            mlflow.log_metric("training_plus_prediction_ms", latency_ms)
            mlflow.log_metric("prediction_ms_per_row", per_row_ms)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            row = {"RunName": run_name, "Model": name}
            row.update(metrics)
            results.append(row)

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model = model
                best_name = run_name

    # Register Best Model
    with mlflow.start_run(run_name=f"registry_{best_name}"):

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            registered_model_name=params["mlflow"]["registered_model_name"]
        )

    # Save Reports
    summary = {
        "best_run": best_name,
        "best_rmse": best_rmse,
        "train_rows": len(X_train),
        "val_rows": len(X_val)
    }

    with open(os.path.join(OUTPUT_DIR, "best_model_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

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
    plt.savefig(
        os.path.join(OUTPUT_DIR, "feature_importance.png"),
        dpi=200
    )
    plt.close()

    logging.info(f"Best model selected: {best_name}")
    logging.info("Training completed successfully.")


# Entry point
if __name__ == "__main__":
    run_training()