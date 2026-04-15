# scripts/preprocess.py

import pandas as pd
import numpy as np
import os
import json
import logging


# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting preprocessing pipeline...")

# Create Directories
RAW_PATH = "/opt/airflow/data/raw"
PROCESSED_PATH = "/opt/airflow/data/processed"
FEATURE_PATH = "/opt/airflow/data/features"

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(FEATURE_PATH, exist_ok=True)


# Load Data
train = pd.read_csv(f"{RAW_PATH}/train.csv")
test = pd.read_csv(f"{RAW_PATH}/test.csv")

logging.info(f"Train shape: {train.shape}")
logging.info(f"Test shape : {test.shape}")


# Convert Date Column
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])


# Remove Duplicates
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)


# Add Flags
train["is_train"] = 1
test["is_train"] = 0

# Add missing columns for merge
train["id"] = np.nan
test["sales"] = np.nan


# Save Clean Raw Data
train.to_csv(
    f"{PROCESSED_PATH}/train_clean.csv",
    index=False
)

test.to_csv(
    f"{PROCESSED_PATH}/test_clean.csv",
    index=False
)

logging.info("Clean raw datasets saved.")


# Combine Train + Test
full = pd.concat([train, test], ignore_index=True)

# Sort for lag features
full = full.sort_values(
    by=["store", "item", "date"]
).reset_index(drop=True)


# Date Features
full["year"] = full["date"].dt.year
full["month"] = full["date"].dt.month
full["day"] = full["date"].dt.day
full["dayofweek"] = full["date"].dt.dayofweek
full["weekofyear"] = full["date"].dt.isocalendar().week.astype(int)
full["quarter"] = full["date"].dt.quarter
full["is_weekend"] = full["dayofweek"].isin([5, 6]).astype(int)


# Cyclical Encoding
full["month_sin"] = np.sin(2 * np.pi * full["month"] / 12)
full["month_cos"] = np.cos(2 * np.pi * full["month"] / 12)

full["dow_sin"] = np.sin(2 * np.pi * full["dayofweek"] / 7)
full["dow_cos"] = np.cos(2 * np.pi * full["dayofweek"] / 7)


# Lag Features
# Only historical train sales used
for lag in [7, 14, 30]:
    full[f"lag_{lag}"] = (
        full.groupby(["store", "item"])["sales"]
        .shift(lag)
    )


# Rolling Means
for window in [7, 30]:
    full[f"rolling_mean_{window}"] = (
        full.groupby(["store", "item"])["sales"]
        .transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    )


# Fill Missing Values
lag_cols = [
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30"
]

for col in lag_cols:
    full[col] = full[col].fillna(0)


# Split Back Safely
train_features = full[
    full["is_train"] == 1
].copy()

test_features = full[
    full["is_train"] == 0
].copy()

# Remove unnecessary columns
test_features.drop(
    columns=["sales"],
    inplace=True
)


# Save Feature Files
train_features.to_csv(
    f"{FEATURE_PATH}/train_features.csv",
    index=False
)

test_features.to_csv(
    f"{FEATURE_PATH}/test_features.csv",
    index=False
)

logging.info("Feature datasets saved.")


# Drift Baseline Stats
stats = {
    "sales_mean": float(train["sales"].mean()),
    "sales_std": float(train["sales"].std()),
    "sales_min": float(train["sales"].min()),
    "sales_max": float(train["sales"].max()),
    "sales_median": float(train["sales"].median())
}

with open(
    f"{PROCESSED_PATH}/baseline_stats.json",
    "w"
) as f:
    json.dump(stats, f, indent=4)

logging.info("Baseline statistics saved.")

logging.info("Preprocessing completed successfully.")