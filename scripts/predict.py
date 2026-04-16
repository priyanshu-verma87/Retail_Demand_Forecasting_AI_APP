# scripts/predict.py
# Purpose:
# - Load best model from MLflow Registry
# - Read engineered test features
# - Generate demand predictions
# - Save predictions 

import os
import logging
import warnings
import pandas as pd
import mlflow.pyfunc

warnings.filterwarnings("ignore")


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# Paths
BASE = "."
DATA_PATH = os.path.join(BASE, "data", "features", "test_features.csv")
OUTPUT_DIR = os.path.join(BASE, "reports")
MLRUNS_DIR = os.path.join(BASE, "mlruns")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)


# MLflow Setup
mlflow.set_tracking_uri("file:./mlruns")

MODEL_URI = "models:/RetailDemandModel/latest"


# Load Test Data
logging.info("Loading engineered test dataset...")

df = pd.read_csv(DATA_PATH)

# Parse date if present
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

logging.info(f"Rows in test set: {len(df)}")


# Save IDs separately
if "id" in df.columns:
    ids = df["id"].copy()
else:
    ids = pd.Series(range(len(df)), name="id")
    

# Drop non-feature columns
drop_cols = []

for col in ["id", "date", "sales", "is_train"]:
    if col in df.columns:
        drop_cols.append(col)

X_test = df.drop(columns=drop_cols)

# Fill remaining NaNs from lag features etc.
X_test = X_test.fillna(0)

logging.info(f"Prediction feature shape: {X_test.shape}")


# Load Registered Model
logging.info("Loading model from MLflow Registry...")

model = mlflow.pyfunc.load_model(MODEL_URI)


# Predict
preds = model.predict(X_test)

# Ensure non-negative demand
preds = [max(0, float(x)) for x in preds]


# Save Predictions
pred_df = pd.DataFrame({
    "id": ids,
    "sales": preds
})

OUT_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
pred_df.to_csv(OUT_PATH, index=False)

logging.info(f"Predictions saved to {OUT_PATH}")
logging.info("Prediction pipeline completed successfully.")