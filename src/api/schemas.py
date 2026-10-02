from datetime import datetime

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Raw transaction submitted for fraud prediction."""

    transaction_id: int = Field(
        ...,
        ge=0,
        description="Unique transaction identifier.",
    )

    customer_id: int = Field(
        ...,
        ge=0,
        description="Customer identifier.",
    )

    terminal_id: int = Field(
        ...,
        ge=0,
        description="Terminal identifier.",
    )

    tx_amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount.",
    )

    tx_datetime: datetime = Field(
        ...,
        description="Transaction timestamp.",
    )


class PredictionResponse(BaseModel):
    """Fraud prediction returned by the model."""

    transaction_id: int

    fraud_probability: float = Field(
        ...,
        ge=0,
        le=1,
    )

    is_fraud: bool

    threshold: float = Field(
        ...,
        ge=0,
        le=1,
    )

    model_name: str


class HealthResponse(BaseModel):
    """API health status."""

    status: str


class ModelInfoResponse(BaseModel):
    """Metadata describing the deployed model."""

    model_name: str
    threshold: float
    feature_count: int