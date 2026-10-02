from functools import lru_cache

import pandas as pd

from fastapi import Depends, FastAPI

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.inference.features import InferenceFeatureBuilder
from src.inference.history import TransactionHistoryStore
from src.inference.predictor import (
    FraudPredictor,
    get_predictor,
)


app = FastAPI(
    title="Fraud Detection ML API",
    description=(
        "Machine learning API for transaction fraud detection "
        "using historical customer and terminal behavior."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_history_store() -> TransactionHistoryStore:
    """
    Return a cached historical transaction store.

    Historical data is loaded once per application process instead of
    being reloaded for every prediction request.
    """

    return TransactionHistoryStore()


@lru_cache(maxsize=1)
def get_feature_builder() -> InferenceFeatureBuilder:
    """Return a cached inference feature builder."""

    return InferenceFeatureBuilder(
        history_store=get_history_store()
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    """Return API health status."""

    return HealthResponse(
        status="ok"
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info(
    predictor: FraudPredictor = Depends(
        get_predictor
    ),
) -> ModelInfoResponse:
    """Return metadata about the deployed fraud model."""

    return ModelInfoResponse(
        model_name=predictor.model_name,
        threshold=predictor.threshold,
        feature_count=len(
            predictor.feature_columns
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(
    transaction: TransactionRequest,
    feature_builder: InferenceFeatureBuilder = Depends(
        get_feature_builder
    ),
    predictor: FraudPredictor = Depends(
        get_predictor
    ),
) -> PredictionResponse:
    """Predict whether a transaction is fraudulent."""

    features = feature_builder.build(
        customer_id=transaction.customer_id,
        terminal_id=transaction.terminal_id,
        tx_amount=transaction.tx_amount,
        tx_datetime=pd.Timestamp(
            transaction.tx_datetime
        ),
    )

    prediction = predictor.predict(
        features
    )

    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=prediction[
            "fraud_probability"
        ],
        is_fraud=prediction[
            "is_fraud"
        ],
        threshold=prediction[
            "threshold"
        ],
        model_name=prediction[
            "model_name"
        ],
    )