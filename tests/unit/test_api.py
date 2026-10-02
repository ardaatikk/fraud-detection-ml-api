from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.main import (
    app,
    get_feature_builder,
)
from src.inference.predictor import get_predictor


client = TestClient(app)


def test_health_endpoint() -> None:
    """Health endpoint should report that the API is running."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_model_info_endpoint() -> None:
    """Model info endpoint should expose deployed model metadata."""

    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_name"] == (
        "hist_gradient_boosting"
    )

    assert body["feature_count"] == 19

    assert 0 <= body["threshold"] <= 1


def test_predict_endpoint() -> None:
    """
    Predict endpoint should connect feature generation
    and model inference correctly.
    """

    feature_builder = MagicMock()

    feature_builder.build.return_value = {
        "TX_AMOUNT": 100.0
    }

    predictor = MagicMock()

    predictor.predict.return_value = {
        "fraud_probability": 0.90,
        "is_fraud": True,
        "threshold": 0.50,
        "model_name": "test_model",
    }

    app.dependency_overrides[
        get_feature_builder
    ] = lambda: feature_builder

    app.dependency_overrides[
        get_predictor
    ] = lambda: predictor

    try:
        response = client.post(
            "/predict",
            json={
                "transaction_id": 123,
                "customer_id": 100,
                "terminal_id": 10,
                "tx_amount": 250.0,
                "tx_datetime": (
                    "2018-09-04T12:00:00"
                ),
            },
        )

        assert response.status_code == 200

        assert response.json() == {
            "transaction_id": 123,
            "fraud_probability": 0.90,
            "is_fraud": True,
            "threshold": 0.50,
            "model_name": "test_model",
        }

    finally:
        app.dependency_overrides.clear()


def test_predict_rejects_missing_fields() -> None:
    """Prediction requests with missing fields should be rejected."""

    response = client.post(
        "/predict",
        json={
            "transaction_id": 123,
            "customer_id": 100,
        },
    )

    assert response.status_code == 422


def test_predict_rejects_negative_amount() -> None:
    """Negative transaction amounts should be rejected."""

    response = client.post(
        "/predict",
        json={
            "transaction_id": 123,
            "customer_id": 100,
            "terminal_id": 10,
            "tx_amount": -50.0,
            "tx_datetime": (
                "2018-09-04T12:00:00"
            ),
        },
    )

    assert response.status_code == 422