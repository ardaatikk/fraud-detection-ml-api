import pandas as pd
import pytest

from src.config.features import MODEL_FEATURES
from src.inference.features import InferenceFeatureBuilder


class FakeHistoryStore:
    """Small in-memory history store for inference feature tests."""

    def __init__(self) -> None:
        self.transactions = pd.DataFrame(
            {
                "TRANSACTION_ID": [
                    1,
                    2,
                    3,
                    4,
                ],
                "TX_DATETIME": pd.to_datetime(
                    [
                        "2018-01-09 12:00:00",
                        "2018-01-09 18:00:00",
                        "2018-01-05 12:00:00",
                        "2017-12-20 12:00:00",
                    ]
                ),
                "CUSTOMER_ID": [
                    100,
                    100,
                    100,
                    100,
                ],
                "TERMINAL_ID": [
                    10,
                    10,
                    10,
                    10,
                ],
                "TX_AMOUNT": [
                    100.0,
                    200.0,
                    50.0,
                    25.0,
                ],
                "TX_FRAUD": [
                    0,
                    1,
                    0,
                    1,
                ],
            }
        )

    def get_customer_history(
        self,
        customer_id: int,
        before: pd.Timestamp,
        days: int,
    ) -> pd.DataFrame:

        start = before - pd.Timedelta(
            days=days
        )

        return self.transactions[
            (self.transactions["CUSTOMER_ID"] == customer_id)
            & (self.transactions["TX_DATETIME"] >= start)
            & (self.transactions["TX_DATETIME"] < before)
        ].copy()

    def get_terminal_history(
        self,
        terminal_id: int,
        before: pd.Timestamp,
        days: int,
    ) -> pd.DataFrame:

        start = before - pd.Timedelta(
            days=days
        )

        return self.transactions[
            (self.transactions["TERMINAL_ID"] == terminal_id)
            & (self.transactions["TX_DATETIME"] >= start)
            & (self.transactions["TX_DATETIME"] < before)
        ].copy()


def build_test_features() -> dict[str, float]:
    """Build inference features for a known synthetic transaction."""

    builder = InferenceFeatureBuilder(
        FakeHistoryStore()
    )

    return builder.build(
        customer_id=100,
        terminal_id=10,
        tx_amount=300.0,
        tx_datetime=pd.Timestamp(
            "2018-01-10 12:00:00"
        ),
    )


def test_inference_builder_produces_model_contract() -> None:
    """Inference builder should produce exactly the trained features."""

    features = build_test_features()

    assert set(features) == set(
        MODEL_FEATURES
    )

    assert len(features) == 19


def test_inference_transaction_features() -> None:
    """Transaction-level features should be derived correctly."""

    features = build_test_features()

    assert features["TX_AMOUNT"] == pytest.approx(
        300.0
    )

    assert features["TX_HOUR"] == 12

    assert features["TX_DAY_OF_WEEK"] == 2


def test_inference_customer_features() -> None:
    """Customer windows should use only historical transactions."""

    features = build_test_features()

    # 1D includes the transaction exactly 24 hours before
    # and the later transaction on the same day.
    assert features[
        "CUSTOMER_TX_COUNT_1D"
    ] == 2

    assert features[
        "CUSTOMER_AVG_AMOUNT_1D"
    ] == pytest.approx(
        150.0
    )

    assert features[
        "AMOUNT_TO_CUSTOMER_AVG_1D"
    ] == pytest.approx(
        2.0
    )

    # 7D additionally contains the Jan 5 transaction.
    assert features[
        "CUSTOMER_TX_COUNT_7D"
    ] == 3

    assert features[
        "CUSTOMER_AVG_AMOUNT_7D"
    ] == pytest.approx(
        350.0 / 3.0
    )


def test_inference_terminal_features() -> None:
    """Terminal activity and fraud history should be calculated correctly."""

    features = build_test_features()

    assert features[
        "TERMINAL_TX_COUNT_1D"
    ] == 2

    assert features[
        "TERMINAL_TX_COUNT_7D"
    ] == 3

    assert features[
        "TERMINAL_TX_COUNT_30D"
    ] == 4

    assert features[
        "TERMINAL_FRAUD_COUNT_7D"
    ] == 1

    assert features[
        "TERMINAL_FRAUD_RATE_7D"
    ] == pytest.approx(
        1 / 3
    )

    assert features[
        "TERMINAL_FRAUD_COUNT_30D"
    ] == 2

    assert features[
        "TERMINAL_FRAUD_RATE_30D"
    ] == pytest.approx(
        0.5
    )


def test_unknown_customer_and_terminal_use_zero_history() -> None:
    """Entities without history should receive zero-valued history features."""

    builder = InferenceFeatureBuilder(
        FakeHistoryStore()
    )

    features = builder.build(
        customer_id=999,
        terminal_id=999,
        tx_amount=100.0,
        tx_datetime=pd.Timestamp(
            "2018-01-10 12:00:00"
        ),
    )

    assert features[
        "CUSTOMER_TX_COUNT_30D"
    ] == 0

    assert features[
        "CUSTOMER_AVG_AMOUNT_30D"
    ] == 0

    assert features[
        "AMOUNT_TO_CUSTOMER_AVG_30D"
    ] == 0

    assert features[
        "TERMINAL_TX_COUNT_30D"
    ] == 0

    assert features[
        "TERMINAL_FRAUD_COUNT_30D"
    ] == 0

    assert features[
        "TERMINAL_FRAUD_RATE_30D"
    ] == 0