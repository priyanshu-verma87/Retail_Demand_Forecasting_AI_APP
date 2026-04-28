# Retail Demand Forecasting AI Application

An end-to-end MLOps project for forecasting retail sales demand using historical store-item data. The system combines data engineering, experiment tracking, model serving, monitoring, CI automation, and a user-friendly frontend.

---

## Project Overview

This application predicts future sales for a given **store**, **item**, and **date**. It is designed as a production-style machine learning system following core MLOps principles:

- Automated data validation and preprocessing
- Reproducible training pipelines using DVC
- Experiment tracking with MLflow
- REST API serving with FastAPI
- Interactive frontend with Streamlit
- Monitoring using Prometheus and Grafana
- Containerized multi-service deployment using Docker Compose
- CI automation using GitHub Actions (self-hosted runner)

---

## Features

### User Features

- Forecast sales demand by entering store ID, item ID, and date
- View prediction latency and recent predictions
- Check API/model health status
- Access monitoring dashboards

### Engineering Features

- Airflow DAG for validation and preprocessing
- DVC pipeline for reproducible ML stages
- MLflow experiment tracking and model registry
- Prometheus metrics exporter instrumentation
- Grafana dashboards for live observability
- Unit and integration tests using pytest

---

## Tech Stack

| Layer                  | Technology              |
| ---------------------- | ----------------------- |
| Frontend               | Streamlit               |
| API Gateway            | FastAPI                 |
| Model Server           | FastAPI + MLflow        |
| Workflow Orchestration | Apache Airflow          |
| Data/Model Versioning  | DVC + Git               |
| Experiment Tracking    | MLflow                  |
| Monitoring             | Prometheus + Grafana    |
| Testing                | pytest                  |
| Containerization       | Docker + Docker Compose |
| CI/CD                  | GitHub Actions          |

---

## Architecture

```text
Streamlit UI
   ↓
FastAPI API
   ↓
Model Server (MLflow model)
   ↓
Prediction Response

Airflow → Validation + Preprocessing
DVC → Training + Evaluation Pipeline
Prometheus → Metrics Scraping
Grafana → Dashboards
```

---

## Repository Structure

```text
.
# Project Folder Structure

```

├── .dvc/
├── .github/
│ └── workflows/
├── app/
├── dags/
├── data/
├── frontend/
├── model_server/
├── monitoring/
├── notebooks/
├── reports/
├── scripts/
├── tests/
│
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.deploy.yaml
├── docker-compose.yaml
├── dvc.lock
├── dvc.yaml
├── params.yaml
├── pytest.ini
└── requirements.txt

````

---

## Setup Instructions

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd <repo-folder>
````

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 3. Pull Versioned Data (DVC)

```bash
dvc pull
```

## 4. Start Deployment Stack

```bash
docker compose -f docker-compose.deploy.yaml up -d --build
```

## 5. Open Services

| Service      | URL                   |
| ------------ | --------------------- |
| Frontend     | http://localhost:8501 |
| API          | http://localhost:8000 |
| Model Server | http://localhost:8001 |
| MLflow       | http://localhost:5000 |
| Grafana      | http://localhost:3000 |
| Prometheus   | http://localhost:9090 |
| Airflow      | http://localhost:8080 |

---

## Running Airflow Pipeline

Start Airflow stack:

```bash
docker compose up -d
```

Trigger DAG:

```bash
airflow dags trigger retail_demand_pipeline
```

Pipeline stages:

- validate_data
- preprocess_data
- generate_baseline

---

## Running Training Pipeline

```bash
dvc repro
```

This executes tracked stages such as:

- train
- predict
- evaluate

---

## API Usage

## Health Check

```bash
GET /health
```

## Prediction Request

```bash
POST /predict
```

Example payload:

```json
{
  "store": 1,
  "item": 1,
  "date": "2018-01-15"
}
```

Example response:

```json
{
  "predicted_sales": 32.47,
  "latency_ms": 18.6
}
```

---

## Monitoring

### Prometheus Metrics

Collected metrics include:

- Total API requests
- Successful requests
- Errors
- Request latency
- Active requests
- Prediction distribution
- Model server health
- Inference latency

### Grafana Dashboards

Dashboards visualize:

- Throughput
- Average latency
- Error rate
- Model readiness
- Prediction trends

---

## Testing

Run unit tests:

```bash
pytest tests/test_api.py tests/test_model_server.py -v
```

Run integration tests:

```bash
pytest tests/test_integration.py -v
```

---

## CI/CD Workflow

GitHub Actions pipeline performs:

1. Checkout code
2. Install dependencies
3. DVC pull
4. Run unit tests
5. Start Airflow
6. Trigger preprocessing DAG
7. Run DVC training pipeline
8. Build deployment containers
9. Run integration tests

---

## Design Principles

- Loose coupling between UI and backend through REST APIs
- Reproducibility through DVC + MLflow
- Containerized environment parity
- Observability-first architecture
- Modular maintainable codebase

---

## Known Limitations

- Designed primarily for local/on-prem execution
- Large retraining workflows may require higher compute resources
- Dataset access depends on configured DVC remote

---

## Author

Priyanshu Verma
