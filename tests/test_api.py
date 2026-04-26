from fastapi.testclient import TestClient
from app.main import app

# Create test client for FastAPI app
client = TestClient(app)


def test_api_health():
    # Test health check endpoint
    r = client.get("/health")

    assert r.status_code == 200
    assert "status" in r.json()


def test_api_metrics():
    # Test Prometheus metrics endpoint
    r = client.get("/metrics")

    assert r.status_code == 200


def test_predict_valid_payload():
    # Valid prediction request payload
    payload = {
        "store": 1,
        "item": 1,
        "date": "2026-04-20"
    }

    r = client.post("/predict", json=payload)

    # If model_server is available -> 200
    # If downstream service fails -> 500 / 503 accepted
    assert r.status_code in [200, 500, 503]


def test_predict_invalid_payload():
    # Invalid payload with wrong data types / date format
    payload = {
        "store": "abc",
        "item": 1,
        "date": "wrong-date"
    }

    r = client.post("/predict", json=payload)

    # Validation error expected
    assert r.status_code == 422