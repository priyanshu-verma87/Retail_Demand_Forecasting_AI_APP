# scripts/validate.py

import pandas as pd
import logging
import os
import sys


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting data validation pipeline...")

# File Paths
TRAIN_PATH = "/opt/airflow/data/raw/train.csv"
TEST_PATH = "/opt/airflow/data/raw/test.csv"


# Check File Existence
if not os.path.exists(TRAIN_PATH):
    logging.error("train.csv not found.")
    sys.exit(1)

if not os.path.exists(TEST_PATH):
    logging.error("test.csv not found.")
    sys.exit(1)

logging.info("Raw data files found.")


# Load Data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

logging.info(f"Train shape: {train.shape}")
logging.info(f"Test shape : {test.shape}")


# Expected Schemas
expected_train_cols = {"date", "store", "item", "sales"}

expected_test_cols = {"id", "date", "store", "item"}


# Schema Validation
if set(train.columns) != expected_train_cols:
    logging.error("Train schema mismatch.")
    sys.exit(1)

if set(test.columns) != expected_test_cols:
    logging.error("Test schema mismatch.")
    sys.exit(1)

logging.info("Schema validation passed.")


# Convert Date Columns
train["date"] = pd.to_datetime(
    train["date"],
    errors="coerce"
)

test["date"] = pd.to_datetime(
    test["date"],
    errors="coerce"
)

if train["date"].isnull().sum() > 0:
    logging.error("Invalid dates found in train.")
    sys.exit(1)

if test["date"].isnull().sum() > 0:
    logging.error("Invalid dates found in test.")
    sys.exit(1)

logging.info("Date validation passed.")


# Null Checks
if train.isnull().sum().sum() > 0:
    logging.error("Null values found in train.")
    sys.exit(1)

if test.isnull().sum().sum() > 0:
    logging.error("Null values found in test.")
    sys.exit(1)

logging.info("Null value check passed.")


# Duplicate Checks
train_duplicates = train.duplicated().sum()
test_duplicates = test.duplicated().sum()

if train_duplicates > 0:
    logging.warning(
        f"{train_duplicates} duplicate rows found in train."
    )
else:
    logging.info("No duplicates in train.")

if test_duplicates > 0:
    logging.warning(
        f"{test_duplicates} duplicate rows found in test."
    )
else:
    logging.info("No duplicates in test.")


# Data Type Checks
numeric_cols_train = ["store", "item", "sales"]
numeric_cols_test = ["id", "store", "item"]

for col in numeric_cols_train:
    if not pd.api.types.is_numeric_dtype(train[col]):
        logging.error(f"{col} must be numeric in train.")
        sys.exit(1)

for col in numeric_cols_test:
    if not pd.api.types.is_numeric_dtype(test[col]):
        logging.error(f"{col} must be numeric in test.")
        sys.exit(1)

logging.info("Data type validation passed.")


# Range Checks
if (train["sales"] < 0).any():
    logging.error("Negative sales found.")
    sys.exit(1)

if (train["store"] <= 0).any():
    logging.error("Invalid store values in train.")
    sys.exit(1)

if (test["store"] <= 0).any():
    logging.error("Invalid store values in test.")
    sys.exit(1)

if (train["item"] <= 0).any():
    logging.error("Invalid item values in train.")
    sys.exit(1)

if (test["item"] <= 0).any():
    logging.error("Invalid item values in test.")
    sys.exit(1)

logging.info("Range checks passed.")


# Final Success
logging.info("Data validation completed successfully.")