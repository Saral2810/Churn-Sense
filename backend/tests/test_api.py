import io

from tests.conftest import HIGH_RISK, LOW_RISK


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"]
    assert "X-Request-ID" in r.headers


def test_predict_high_vs_low_risk(client):
    hi = client.post("/api/v1/predict", json=HIGH_RISK).json()
    lo = client.post("/api/v1/predict", json=LOW_RISK).json()
    assert hi["churn_probability"] > 0.5 > lo["churn_probability"]
    assert hi["will_churn"] and not lo["will_churn"]
    assert hi["risk_tier"] in {"high", "critical"}
    assert hi["drivers"] and hi["recommendations"]
    contrib = {d["feature"]: d["contribution"] for d in hi["drivers"]}
    # Overlapping churn rules must not cancel out: each risky behaviour pushes risk up.
    assert contrib.get("contract_length", 0) > 0
    assert contrib.get("support_calls", 0) > 0


def test_predict_validation_error(client):
    r = client.post("/api/v1/predict", json={**HIGH_RISK, "contract_length": "Weekly"})
    assert r.status_code == 422
    r = client.post("/api/v1/predict", json={**HIGH_RISK, "age": -5})
    assert r.status_code == 422


def test_batch(client):
    r = client.post("/api/v1/predict/batch", json={"customers": [HIGH_RISK, LOW_RISK]})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["scored"] == 2
    assert body["summary"]["predicted_churners"] == 1


def test_csv_upload_with_original_headers_and_bad_row(client):
    csv_text = (
        "CustomerID,Age,Gender,Tenure,Usage Frequency,Support Calls,Payment Delay,"
        "Subscription Type,Contract Length,Total Spend,Last Interaction\n"
        "101,42,Female,8,6,7,24,Basic,Monthly,320,21\n"
        "102,30,male,48,25,1,3,Premium,Annual,900,3\n"
        "103,abc,Male,48,25,1,3,Premium,Annual,900,3\n"
    )
    files = {"file": ("customers.csv", io.BytesIO(csv_text.encode()), "text/csv")}
    body = client.post("/api/v1/predict/csv", files=files).json()
    assert body["summary"] == {**body["summary"], "rows": 3, "scored": 2, "failed": 1}
    assert body["results"][0]["customer_id"] == "101"
    assert body["errors"][0]["row"] == 2


def test_roi(client):
    body = client.get("/api/v1/roi", params={"retention_cost": 300, "revenue_saved": 2000, "threshold": 0.5}).json()
    assert body["cohort_size"] > 1000
    assert len(body["curve"]) == 99
    assert body["net_profit"] == round(body["revenue_protected"] - body["campaign_cost"], 2)


def test_analytics_and_model_card(client):
    a = client.get("/api/v1/analytics").json()
    assert {"kpis", "segments", "insights", "by_contract"} <= set(a)
    m = client.get("/api/v1/model").json()
    assert m["holdout"]["roc_auc"] > 0.9
    assert m["feature_importance"]


def test_customers_filter(client):
    body = client.get("/api/v1/customers", params={"tier": "critical", "limit": 5}).json()
    assert len(body["items"]) <= 5
    assert all(c["risk_tier"] == "critical" for c in body["items"])


def test_api_key_enforced(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.post("/api/v1/predict", json=HIGH_RISK).status_code == 401
    ok = client.post("/api/v1/predict", json=HIGH_RISK, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
