# scripts/evaluate.py
# Purpose:
# - Read outputs from train.py / predict.py
# - Summarize best model performance
# - Create comparison chart
# - Generate final evaluation artifacts for DVC / report

import os
import json
import logging
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# Paths
BASE = "."
REPORT_DIR = os.path.join(BASE, "reports")

MODEL_COMPARE_PATH = os.path.join(REPORT_DIR, "model_comparison.csv")
BEST_SUMMARY_PATH = os.path.join(REPORT_DIR, "best_model_summary.json")
FINAL_METRICS_PATH = os.path.join(REPORT_DIR, "final_metrics.json")
TEXT_REPORT_PATH = os.path.join(REPORT_DIR, "evaluation_report.txt")
CHART_PATH = os.path.join(REPORT_DIR, "model_rmse_comparison.png")

os.makedirs(REPORT_DIR, exist_ok=True)


# Load model comparison
logging.info("Loading model comparison report...")

df = pd.read_csv(MODEL_COMPARE_PATH)

# Ensure sorted
df = df.sort_values("RMSE").reset_index(drop=True)

best_row = df.iloc[0]


# Save final metrics json
final_metrics = {
    "best_model": str(best_row["Model"]),
    "best_run": str(best_row["RunName"]),
    "MAE": float(best_row["MAE"]),
    "RMSE": float(best_row["RMSE"]),
    "R2": float(best_row["R2"]),
    "MAPE": float(best_row["MAPE"])
}

with open(FINAL_METRICS_PATH, "w") as f:
    json.dump(final_metrics, f, indent=4)

logging.info("final_metrics.json saved.")


# Create text report
lines = []
lines.append("Retail Demand Forecasting - Final Evaluation Report")
lines.append("=" * 55)
lines.append("")
lines.append(f"Best Model : {best_row['Model']}")
lines.append(f"Best Run   : {best_row['RunName']}")
lines.append("")
lines.append("Performance Metrics")
lines.append("-" * 25)
lines.append(f"MAE   : {best_row['MAE']:.4f}")
lines.append(f"RMSE  : {best_row['RMSE']:.4f}")
lines.append(f"R2    : {best_row['R2']:.4f}")
lines.append(f"MAPE  : {best_row['MAPE']:.4f}%")
lines.append("")
lines.append("Models Ranked by RMSE")
lines.append("-" * 25)

for i, row in df.iterrows():
    lines.append(
        f"{i+1}. {row['Model']} | "
        f"RMSE={row['RMSE']:.4f} | "
        f"MAE={row['MAE']:.4f}"
    )

with open(TEXT_REPORT_PATH, "w") as f:
    f.write("\n".join(lines))

logging.info("evaluation_report.txt saved.")


# Create RMSE comparison chart
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["RMSE"])
plt.title("Model Comparison by RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200)
plt.close()

logging.info("model_rmse_comparison.png saved.")


# read best summary if exists
if os.path.exists(BEST_SUMMARY_PATH):
    logging.info("best_model_summary.json found.")

# Done
logging.info("Evaluation stage completed successfully.")