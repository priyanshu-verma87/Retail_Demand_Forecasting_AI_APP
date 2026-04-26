import math
import time
import logging
import warnings
import pandas as pd
import mlflow
import mlflow.pyfunc
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Model Server Metrics

MODEL_REQUESTS = Counter(
    "model_inference_total",
    "Total inference requests"
)

MODEL_ERRORS = Counter(
    "model_inference_errors_total",
    "Total inference failures"
)

MODEL_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model prediction latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5)
)

MODEL_LOADED = Gauge(
    "model_loaded_status",
    "1 if model loaded else 0"
)

MODEL_OUTPUT = Histogram(
    "model_predicted_sales",
    "Distribution of model outputs",
    buckets=(0, 10, 20, 40, 60, 80, 100, 150, 250)
)

app = FastAPI(
    title="Retail Demand Model Server",
    version="1.0.0"
)
Instrumentator().instrument(app).expose(app)

MODEL_URI = "models:/RetailDemandModel/latest"
model = None


class PredictRequest(BaseModel):
    store: int
    item: int
    date: str


@app.on_event("startup")
def startup():

    global model

    try:
        mlflow.set_tracking_uri("file:./mlruns")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        MODEL_LOADED.set(1)

        logging.info("Model loaded successfully.")
    except Exception as e:
        MODEL_LOADED.set(0)

        logging.error(f"Model load failed: {e}")
        model = None


@app.get("/health")
def health():

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "healthy"}


@app.get("/ready")
def ready():

    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    return {"status": "ready"}


def build_features(store, item, date_str):

    dt = pd.to_datetime(date_str)

    month = dt.month
    day = dt.day
    dow = dt.dayofweek
    week = int(dt.isocalendar().week)
    quarter = dt.quarter

    row = {
        "store": store,
        "item": item,
        "year": dt.year,
        "month": month,
        "day": day,
        "dayofweek": dow,
        "weekofyear": week,
        "quarter": quarter,
        "is_weekend": 1 if dow >= 5 else 0,

        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),

        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),

        "lag_7": 0,
        "lag_14": 0,
        "lag_30": 0,
        "rolling_mean_7": 0,
        "rolling_mean_30": 0
    }

    cols = [
        "store","item","year","month","day","dayofweek",
        "weekofyear","quarter","is_weekend",
        "month_sin","month_cos","dow_sin","dow_cos",
        "lag_7","lag_14","lag_30",
        "rolling_mean_7","rolling_mean_30"
    ]

    return pd.DataFrame([row])[cols]


@app.post("/infer")
def infer(req: PredictRequest):

    global model

    MODEL_REQUESTS.inc()

    if model is None:
        MODEL_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        start = time.perf_counter()

        X = build_features(req.store, req.item, req.date)
        pred = float(model.predict(X)[0])

        end = time.perf_counter()

        MODEL_LATENCY.observe(end - start)
        pred = max(0, pred)
        MODEL_OUTPUT.observe(pred)
        return {
            "store": req.store,
            "item": req.item,
            "date": req.date,
            "predicted_sales": round(max(0, pred), 2)
        }

    except Exception as e:
        MODEL_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))
    
