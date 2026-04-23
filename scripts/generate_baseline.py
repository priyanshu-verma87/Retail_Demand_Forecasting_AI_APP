import pandas as pd
import json
import os
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
FEATURE_PATH = "/opt/airflow/data/features"
PROCESSED_PATH = "/opt/airflow/data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

df = pd.read_csv(f"{FEATURE_PATH}/train_features.csv")

# Stats of train data
stats = {
    "month_mean": float(df["month"].mean()),
    "month_std": float(df["month"].std()),
    "dayofweek_mean": float(df["dayofweek"].mean()),
    "is_weekend_rate": float(df["is_weekend"].mean()),
    "store_unique": int(df["store"].nunique()),
    "item_unique": int(df["item"].nunique())
}

# Save the stats
with open(f"{PROCESSED_PATH}/baseline_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

logging.info("Feature baseline stats saved successfully.")