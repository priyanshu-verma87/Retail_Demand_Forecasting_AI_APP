import os
import time
import sqlite3
import logging
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest

DB_PATH = "reports/predictions.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(
    title="Retail Demand Forecast API",
    version="1.0.0"
)

def init_db():
    os.makedirs("reports", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store INTEGER,
            item INTEGER,
            date TEXT,
            month INTEGER,
            dayofweek INTEGER,
            is_weekend INTEGER,
            predicted_sales REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

init_db()

MODEL_SERVER_URL = "http://model_server:8001/infer"

REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
SUCCESS_COUNT = Counter("api_success_total","Total successful prediction requests")
ERROR_COUNT = Counter("api_errors_total", "Total API Errors")
ACTIVE_REQUESTS = Gauge("api_active_requests","Current active API requests")
LATENCY = Histogram("api_latency_seconds","Total API request latency",buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)
PREDICTED_SALES = Histogram("api_predicted_sales","Distribution of predicted sales",buckets=(0, 10, 20, 40, 60, 80, 100, 150, 250))

class PredictRequest(BaseModel):
    store: int
    item: int
    date: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    try:
        r = requests.get("http://model_server:8001/health", timeout=2)
        if r.status_code == 200:
            return {"status": "ready"}
    except:
        pass

    raise HTTPException(status_code=503, detail="Model server unavailable")


@app.post("/predict")
def predict(req: PredictRequest):

    REQUEST_COUNT.inc()
    ACTIVE_REQUESTS.inc()

    try:
        start = time.perf_counter()

        r = requests.post(
            MODEL_SERVER_URL,
            json=req.model_dump(),
            timeout=10
        )

        if r.status_code != 200:
            ERROR_COUNT.inc()
            raise HTTPException(status_code=500, detail="Inference failed")

        end = time.perf_counter()

        latency_ms = round((end - start) * 1000, 2)

        LATENCY.observe(end - start)

        data = r.json()
        data["latency_ms"] = latency_ms

        SUCCESS_COUNT.inc()

        if "predicted_sales" in data:
            PREDICTED_SALES.observe(
                float(data["predicted_sales"])
            )

        # Save recent prediction log
        os.makedirs("reports", exist_ok=True)

        dt = pd.to_datetime(req.date)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO prediction_logs
        (store, item, date, month, dayofweek, is_weekend, predicted_sales)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            req.store,
            req.item,
            req.date,
            int(dt.month),
            int(dt.dayofweek),
            1 if dt.dayofweek >= 5 else 0,
            float(data["predicted_sales"])
        ))

        conn.commit()
        conn.close()

        return data

    except Exception as e:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/recent_predictions")
def recent_predictions():
    conn = sqlite3.connect("reports/predictions.db")

    df = pd.read_sql_query("""
        SELECT created_at, store, item, date, predicted_sales
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT 10
    """, conn)

    conn.close()

    return df.to_dict(orient="records")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")