import pytest
from fastapi.testclient import TestClient

from app.main import app

HIGH_RISK = {
    "age": 42,
    "gender": "Female",
    "tenure": 8,
    "usage_frequency": 6,
    "support_calls": 7,
    "payment_delay": 24,
    "subscription_type": "Basic",
    "contract_length": "Monthly",
    "total_spend": 320,
    "last_interaction": 21,
}
LOW_RISK = {
    "age": 30,
    "gender": "Male",
    "tenure": 48,
    "usage_frequency": 25,
    "support_calls": 1,
    "payment_delay": 3,
    "subscription_type": "Premium",
    "contract_length": "Annual",
    "total_spend": 900,
    "last_interaction": 3,
}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
