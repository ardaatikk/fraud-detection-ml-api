import pandas as pd
import pytest

from src.data.validate import validate_dataset


def make_valid_dataset() -> pd.DataFrame:
    """Create a small valid transaction dataset for testing."""

    return pd.DataFrame(
        {
            "TRANSACTION_ID": [
                1,
                2,
                3,
            ],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 10:00:00",
                    "2018-04-01 11:00:00",
                    "2018-04-01 12:00:00",
                ]
            ),
            "CUSTOMER_ID": [
                100,
                101,
                102,
            ],
            "TERMINAL_ID": [
                10,
                11,
                12,
            ],
            "TX_AMOUNT": [
                50.0,
                100.0,
                25.0,
            ],
            "TX_TIME_SECONDS": [
                36000,
                39600,
                43200,
            ],
            "TX_TIME_DAYS": [
                0,
                0,
                0,
            ],
            "TX_FRAUD": [
                0,
                1,
                0,
            ],
            "TX_FRAUD_SCENARIO": [
                0,
                1,
                0,
            ],
        }
    )


def test_valid_dataset_passes() -> None:
    """A valid dataset should pass validation."""

    df = make_valid_dataset()

    validate_dataset(df)


def test_missing_required_column_raises_error() -> None:
    """Missing required columns should be rejected."""

    df = make_valid_dataset().drop(
        columns=["TX_AMOUNT"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_dataset(df)


def test_empty_dataset_raises_error() -> None:
    """An empty dataset should be rejected."""

    df = make_valid_dataset().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="Dataset is empty",
    ):
        validate_dataset(df)


def test_missing_transaction_id_raises_error() -> None:
    """Missing transaction IDs should be rejected."""

    df = make_valid_dataset()

    df.loc[
        0,
        "TRANSACTION_ID",
    ] = None

    with pytest.raises(
        ValueError,
        match="TRANSACTION_ID contains missing values",
    ):
        validate_dataset(df)


def test_duplicate_transaction_id_raises_error() -> None:
    """Duplicate transaction IDs should be rejected."""

    df = make_valid_dataset()

    df.loc[
        1,
        "TRANSACTION_ID",
    ] = df.loc[
        0,
        "TRANSACTION_ID",
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate TRANSACTION_ID",
    ):
        validate_dataset(df)


def test_missing_target_raises_error() -> None:
    """Missing target values should be rejected."""

    df = make_valid_dataset()

    df.loc[
        0,
        "TX_FRAUD",
    ] = None

    with pytest.raises(
        ValueError,
        match="TX_FRAUD contains missing values",
    ):
        validate_dataset(df)


def test_invalid_target_raises_error() -> None:
    """Target values other than 0 and 1 should be rejected."""

    df = make_valid_dataset()

    df.loc[
        0,
        "TX_FRAUD",
    ] = 2

    with pytest.raises(
        ValueError,
        match="Invalid TX_FRAUD values",
    ):
        validate_dataset(df)


def test_missing_transaction_amount_raises_error() -> None:
    """Missing transaction amounts should be rejected."""

    df = make_valid_dataset()

    df.loc[
        0,
        "TX_AMOUNT",
    ] = None

    with pytest.raises(
        ValueError,
        match="TX_AMOUNT contains missing values",
    ):
        validate_dataset(df)


def test_negative_transaction_amount_raises_error() -> None:
    """Negative transaction amounts should be rejected."""

    df = make_valid_dataset()

    df.loc[
        0,
        "TX_AMOUNT",
    ] = -10.0

    with pytest.raises(
        ValueError,
        match="Negative transaction amounts",
    ):
        validate_dataset(df)


def test_missing_datetime_raises_error() -> None:
    """Missing transaction datetimes should be rejected."""

    df = make_valid_dataset()

    df.loc[
        0,
        "TX_DATETIME",
    ] = pd.NaT

    with pytest.raises(
        ValueError,
        match="TX_DATETIME contains missing values",
    ):
        validate_dataset(df)