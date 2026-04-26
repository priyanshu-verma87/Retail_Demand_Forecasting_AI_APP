# scripts/check_drift_retrain.py

import os
import json
import sqlite3
import logging
import pandas as pd
import yaml
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

BASELINE_PATH = "data/processed/baseline_stats.json"
DB_PATH = "reports/predictions.db"
PARAMS_PATH = "params.yaml"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE_FILE = os.path.join(BASE_DIR, "docker-compose.deploy.yaml")


def load_threshold():
    try:
        with open(PARAMS_PATH, "r") as f:
            params = yaml.safe_load(f)
        return float(params["drift"]["threshold_percent"])
    except:
        return 20.0


def pct_change(current, baseline, eps=1e-6):
    return abs(current - baseline) / max(abs(baseline), eps) * 100


def main():

    os.makedirs("reports", exist_ok=True)

    if not os.path.exists(BASELINE_PATH):
        logging.error("Baseline file not found.")
        return

    if not os.path.exists(DB_PATH):
        logging.error("Predictions database not found.")
        return

    with open(BASELINE_PATH, "r") as f:
        baseline = json.load(f)

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
    SELECT *
    FROM prediction_logs
    ORDER BY id DESC
    LIMIT 100
    """, conn)

    conn.close()

    if len(df) < 30:
        logging.warning("Need at least 30 rows for drift detection.")
        return

    threshold = load_threshold()

    result = {
        "rows_checked": int(len(df)),
        "drift_detected": False,
        "reasons": []
    }

    # Month Mean Drift
    if "month" in df.columns:
        current = float(df["month"].mean())
        base = float(baseline["month_mean"])

        drift = pct_change(current, base)

        if drift > threshold:
            result["drift_detected"] = True
            result["reasons"].append(
                f"month_mean shifted {drift:.2f}%"
            )

    # DayOfWeek Mean Drift
    if "dayofweek" in df.columns:
        current = float(df["dayofweek"].mean())
        base = float(baseline["dayofweek_mean"])

        drift = pct_change(current, base)

        if drift > threshold:
            result["drift_detected"] = True
            result["reasons"].append(
                f"dayofweek_mean shifted {drift:.2f}%"
            )

    # Weekend Rate Drift
    if "is_weekend" in df.columns:
        current = float(df["is_weekend"].mean())
        base = float(baseline["is_weekend_rate"])

        drift = pct_change(current, base)

        if drift > threshold:
            result["drift_detected"] = True
            result["reasons"].append(
                f"is_weekend_rate shifted {drift:.2f}%"
            )

    # Prediction Summary
    if "predicted_sales" in df.columns:
        result["prediction_mean"] = float(df["predicted_sales"].mean())
        result["prediction_std"] = float(df["predicted_sales"].std())

    logging.info(result)

    if result["drift_detected"]:

        logging.info("Drift detected.")
        logging.info("Reasons: %s", result["reasons"])
        logging.info("Starting retraining...")

        try:
            subprocess.run(
                [
                    "docker", "compose",
                    "-f", COMPOSE_FILE,
                    "run", "--rm",
                    "model_server",
                    "python", "scripts/train.py"
                ],
                check=True
            )

            subprocess.run(
                [
                    "docker", "compose",
                    "-f", COMPOSE_FILE,
                    "restart", "model_server"
                ],
                check=True
            )

            logging.info("Retraining completed successfully.")

        except Exception as e:
            logging.error("Retraining failed: %s", str(e))

    else:
        logging.info("No significant drift. Retraining skipped.")


if __name__ == "__main__":
    main()