import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_customer_features,
    add_terminal_features,
    add_transaction_features,
    build_features,
    clean_engineered_features,
    validate_engineered_features,
)


def make_transaction_history() -> pd.DataFrame:
    """Create deterministic transaction history for feature tests."""

    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3, 4],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 10:00:00",
                    "2018-04-01 12:00:00",
                    "2018-04-02 10:00:00",
                    "2018-04-03 10:00:00",
                ]
            ),
            "CUSTOMER_ID": [
                100,
                100,
                100,
                200,
            ],
            "TERMINAL_ID": [
                10,
                10,
                10,
                20,
            ],
            "TX_AMOUNT": [
                100.0,
                200.0,
                300.0,
                50.0,
            ],
            "TX_FRAUD": [
                0,
                1,
                0,
                0,
            ],
        }
    )


def test_transaction_features_are_created_correctly() -> None:
    """Hour and day-of-week features should match TX_DATETIME."""

    df = make_transaction_history()

    result = add_transaction_features(df)

    assert result["TX_HOUR"].tolist() == [
        10,
        12,
        10,
        10,
    ]

    # 2018-04-01 = Sunday = 6
    # 2018-04-02 = Monday = 0
    # 2018-04-03 = Tuesday = 1
    assert result["TX_DAY_OF_WEEK"].tolist() == [
        6,
        6,
        0,
        1,
    ]


def test_customer_history_excludes_current_transaction() -> None:
    """
    Customer rolling features must use only transactions that occurred
    before the current transaction.
    """

    df = make_transaction_history()

    result = add_customer_features(df)
    result = clean_engineered_features(result)

    customer_rows = (
        result[result["CUSTOMER_ID"] == 100]
        .sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )

    first = customer_rows.iloc[0]
    second = customer_rows.iloc[1]
    third = customer_rows.iloc[2]

    # First transaction has no previous customer history.
    assert first["CUSTOMER_TX_COUNT_1D"] == 0

    # Second transaction should see only the first transaction.
    assert second["CUSTOMER_TX_COUNT_1D"] == 1
    assert second["CUSTOMER_AVG_AMOUNT_1D"] == pytest.approx(
        100.0
    )

    # The current amount (200) must not be included in its own average.
    assert second["CUSTOMER_AVG_AMOUNT_1D"] != pytest.approx(
        150.0
    )

    # The 1-day window includes transactions exactly 24 hours
    # before the current transaction, while excluding the current
    # transaction itself because the right boundary is open.
    assert third["CUSTOMER_TX_COUNT_1D"] == 2
    assert third["CUSTOMER_AVG_AMOUNT_1D"] == pytest.approx(
        150.0
    )


def test_customer_amount_ratio_uses_historical_average() -> None:
    """Amount ratio should compare current amount with past behavior."""

    df = make_transaction_history()

    result = add_customer_features(df)

    second = (
        result[result["CUSTOMER_ID"] == 100]
        .sort_values("TX_DATETIME")
        .iloc[1]
    )

    # Current amount = 200
    # Historical 1D average = 100
    assert second[
        "AMOUNT_TO_CUSTOMER_AVG_1D"
    ] == pytest.approx(2.0)


def test_terminal_history_excludes_current_transaction() -> None:
    """
    Terminal statistics must not use the current transaction's label.
    """

    df = make_transaction_history()

    result = add_terminal_features(df)
    result = clean_engineered_features(result)

    terminal_rows = (
        result[result["TERMINAL_ID"] == 10]
        .sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )

    first = terminal_rows.iloc[0]
    second = terminal_rows.iloc[1]
    third = terminal_rows.iloc[2]

    # No terminal history exists before the first transaction.
    assert first["TERMINAL_TX_COUNT_7D"] == 0
    assert first["TERMINAL_FRAUD_COUNT_7D"] == 0

    # Second transaction itself is fraud, but it must not count itself.
    assert second["TX_FRAUD"] == 1
    assert second["TERMINAL_TX_COUNT_7D"] == 1
    assert second["TERMINAL_FRAUD_COUNT_7D"] == 0
    assert second["TERMINAL_FRAUD_RATE_7D"] == pytest.approx(
        0.0
    )

    # Third transaction should now see both previous transactions,
    # including the previous fraud.
    assert third["TERMINAL_TX_COUNT_7D"] == 2
    assert third["TERMINAL_FRAUD_COUNT_7D"] == 1
    assert third["TERMINAL_FRAUD_RATE_7D"] == pytest.approx(
        0.5
    )


def test_clean_engineered_features_removes_nan_and_inf() -> None:
    """Cleaning should remove undefined engineered feature values."""

    df = pd.DataFrame(
        {
            "CUSTOMER_TX_COUNT_1D": [
                np.nan,
                2.0,
            ],
            "CUSTOMER_AVG_AMOUNT_1D": [
                np.nan,
                50.0,
            ],
            "AMOUNT_TO_CUSTOMER_AVG_1D": [
                np.inf,
                2.0,
            ],
            "TERMINAL_FRAUD_COUNT_7D": [
                np.nan,
                1.0,
            ],
            "TERMINAL_FRAUD_RATE_7D": [
                np.nan,
                0.5,
            ],
        }
    )

    result = clean_engineered_features(df)

    assert not result.isna().any().any()

    numeric_values = result.to_numpy()

    assert np.isfinite(numeric_values).all()

    assert result.loc[
        0,
        "CUSTOMER_TX_COUNT_1D",
    ] == 0

    assert result.loc[
        0,
        "AMOUNT_TO_CUSTOMER_AVG_1D",
    ] == 0


def test_build_features_returns_chronological_data() -> None:
    """Full feature pipeline should restore chronological ordering."""

    df = make_transaction_history().iloc[
        [3, 1, 0, 2]
    ].copy()

    result = build_features(df)

    assert result["TX_DATETIME"].is_monotonic_increasing


def test_build_features_produces_no_missing_values() -> None:
    """Full feature pipeline should produce clean engineered features."""

    df = make_transaction_history()

    result = build_features(df)

    engineered_columns = [
        column
        for column in result.columns
        if (
            column.startswith("CUSTOMER_")
            or column.startswith("TERMINAL_")
            or column.startswith("AMOUNT_TO_")
            or column in {
                "TX_HOUR",
                "TX_DAY_OF_WEEK",
            }
        )
    ]

    # CUSTOMER_ID and TERMINAL_ID are included by the prefix checks,
    # which is fine because they are also expected to be complete.
    assert not result[
        engineered_columns
    ].isna().any().any()


def test_engineered_validation_accepts_valid_features() -> None:
    """Correctly engineered data should pass validation."""

    df = make_transaction_history()

    result = build_features(df)

    validate_engineered_features(result)