import pandas as pd

from src.config.settings import TARGET_COLUMN


REQUIRED_COLUMNS = {
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    TARGET_COLUMN,
    "TX_FRAUD_SCENARIO",
}


def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """Validate the raw transaction dataset."""

    print(
        "Validating dataset..."
    )

    # --------------------------------------------------------------
    # Required columns
    # --------------------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------------

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    # --------------------------------------------------------------
    # Transaction IDs
    # --------------------------------------------------------------

    if df[
        "TRANSACTION_ID"
    ].isna().any():
        raise ValueError(
            "TRANSACTION_ID contains "
            "missing values."
        )

    if df[
        "TRANSACTION_ID"
    ].duplicated().any():
        raise ValueError(
            "Duplicate TRANSACTION_ID "
            "values detected."
        )

    # --------------------------------------------------------------
    # Target
    # --------------------------------------------------------------

    if df[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            f"{TARGET_COLUMN} contains "
            "missing values."
        )

    invalid_targets = (
        set(
            df[
                TARGET_COLUMN
            ].unique()
        )
        - {0, 1}
    )

    if invalid_targets:
        raise ValueError(
            f"Invalid {TARGET_COLUMN} "
            "values detected: "
            f"{invalid_targets}"
        )

    # --------------------------------------------------------------
    # Transaction amount
    # --------------------------------------------------------------

    if df[
        "TX_AMOUNT"
    ].isna().any():
        raise ValueError(
            "TX_AMOUNT contains "
            "missing values."
        )

    if (
        df["TX_AMOUNT"]
        < 0
    ).any():
        raise ValueError(
            "Negative transaction "
            "amounts detected."
        )

    # --------------------------------------------------------------
    # Transaction datetime
    # --------------------------------------------------------------

    if df[
        "TX_DATETIME"
    ].isna().any():
        raise ValueError(
            "TX_DATETIME contains "
            "missing values."
        )

    print(
        "Dataset validation passed."
    )