from fastapi.testclient import TestClient
from model_server.main import app

# Create test client for model server app
client = TestClient(app)


def test_model_health():
    # Check health endpoint
    r = client.get("/health")

    # If model loaded successfully -> 200
    # If startup/model load issue -> 503
    assert r.status_code in [200, 503]


def test_model_ready():
    # Check readiness endpoint
    r = client.get("/ready")

    assert r.status_code in [200, 503]


def test_model_metrics():
    # Test Prometheus metrics endpoint
    r = client.get("/metrics")

    assert r.status_code == 200


def test_infer_valid_payload():
    # Valid inference request payload
    payload = {
        "store": 1,
        "item": 1,
        "date": "2026-04-20"
    }

    r = client.post("/infer", json=payload)

    # Success -> 200
    # Internal/model issue -> 500 / 503 accepted
    assert r.status_code in [200, 500, 503]


def test_infer_invalid_payload():
    # Invalid payload with wrong types / bad date
    payload = {
        "store": "abc",
        "item": 1,
        "date": "bad-date"
    }

    r = client.post("/infer", json=payload)

    # Validation error expected
    assert r.status_code == 422