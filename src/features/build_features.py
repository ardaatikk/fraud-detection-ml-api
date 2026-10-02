import numpy as np
import pandas as pd

from src.config.features import (
    ENGINEERED_FEATURES,
)
from src.config.paths import (
    FEATURE_DATA_FILE,
    RAW_DATA_FILE,
)
from src.config.settings import (
    TARGET_COLUMN,
)


CUSTOMER_WINDOWS = {
    "1D": "1d",
    "7D": "7d",
    "30D": "30d",
}

TERMINAL_COUNT_WINDOWS = {
    "1D": "1d",
    "7D": "7d",
    "30D": "30d",
}

TERMINAL_FRAUD_WINDOWS = {
    "7D": "7d",
    "30D": "30d",
}


def add_transaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction-level temporal features."""

    df = df.copy()

    df["TX_HOUR"] = (
        df["TX_DATETIME"]
        .dt.hour
    )

    df["TX_DAY_OF_WEEK"] = (
        df["TX_DATETIME"]
        .dt.dayofweek
    )

    return df


def add_customer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-free historical customer features.

    For every transaction, statistics are calculated
    using only transactions that occurred before the
    current transaction.
    """

    df = df.sort_values(
        [
            "CUSTOMER_ID",
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    ).copy()

    for (
        suffix,
        window,
    ) in CUSTOMER_WINDOWS.items():

        rolling = (
            df
            .set_index(
                "TX_DATETIME"
            )
            .groupby(
                "CUSTOMER_ID"
            )["TX_AMOUNT"]
            .rolling(
                window,
                closed="left",
            )
        )

        count_feature = (
            f"CUSTOMER_TX_COUNT_{suffix}"
        )

        avg_feature = (
            f"CUSTOMER_AVG_AMOUNT_{suffix}"
        )

        ratio_feature = (
            f"AMOUNT_TO_CUSTOMER_AVG_{suffix}"
        )

        df[count_feature] = (
            rolling
            .count()
            .reset_index(
                level=0,
                drop=True,
            )
            .to_numpy()
        )

        df[avg_feature] = (
            rolling
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
            .to_numpy()
        )

        df[ratio_feature] = (
            df["TX_AMOUNT"]
            / df[avg_feature]
        )

    return df


def add_terminal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create leakage-free historical terminal features.

    Transaction activity and fraud statistics are
    calculated using only terminal observations that
    occurred before the current transaction.
    """

    df = df.sort_values(
        [
            "TERMINAL_ID",
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    ).copy()

    # Historical terminal transaction counts
    for (
        suffix,
        window,
    ) in TERMINAL_COUNT_WINDOWS.items():

        rolling_count = (
            df
            .set_index(
                "TX_DATETIME"
            )
            .groupby(
                "TERMINAL_ID"
            )["TRANSACTION_ID"]
            .rolling(
                window,
                closed="left",
            )
            .count()
        )

        feature_name = (
            f"TERMINAL_TX_COUNT_{suffix}"
        )

        df[feature_name] = (
            rolling_count
            .reset_index(
                level=0,
                drop=True,
            )
            .to_numpy()
        )

    # Historical terminal fraud statistics
    for (
        suffix,
        window,
    ) in TERMINAL_FRAUD_WINDOWS.items():

        rolling_fraud = (
            df
            .set_index(
                "TX_DATETIME"
            )
            .groupby(
                "TERMINAL_ID"
            )[TARGET_COLUMN]
            .rolling(
                window,
                closed="left",
            )
        )

        fraud_count_feature = (
            f"TERMINAL_FRAUD_COUNT_{suffix}"
        )

        fraud_rate_feature = (
            f"TERMINAL_FRAUD_RATE_{suffix}"
        )

        df[fraud_count_feature] = (
            rolling_fraud
            .sum()
            .reset_index(
                level=0,
                drop=True,
            )
            .to_numpy()
        )

        df[fraud_rate_feature] = (
            rolling_fraud
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
            .to_numpy()
        )

    return df


def clean_engineered_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle undefined values created by historical
    features.

    Early transactions may have no customer or terminal
    history. Historical counts, averages, ratios and
    fraud statistics are represented as zero when no
    historical information is available.
    """

    count_columns = [
        column
        for column in df.columns
        if "_TX_COUNT_" in column
    ]

    average_columns = [
        column
        for column in df.columns
        if "_AVG_AMOUNT_" in column
    ]

    ratio_columns = [
        column
        for column in df.columns
        if "AMOUNT_TO_" in column
    ]

    fraud_columns = [
        column
        for column in df.columns
        if "TERMINAL_FRAUD_" in column
    ]

    df[count_columns] = (
        df[count_columns]
        .fillna(0)
    )

    df[average_columns] = (
        df[average_columns]
        .fillna(0)
    )

    df[ratio_columns] = (
        df[ratio_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
    )

    df[fraud_columns] = (
        df[fraud_columns]
        .fillna(0)
    )

    return df


def validate_engineered_features(
    df: pd.DataFrame,
) -> None:
    """Validate engineered features before saving."""

    print(
        "Validating engineered features..."
    )

    missing_columns = [
        column
        for column in ENGINEERED_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing engineered features: "
            f"{missing_columns}"
        )

    if (
        df[ENGINEERED_FEATURES]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Engineered features contain "
            "missing values."
        )

    numeric_values = (
        df[ENGINEERED_FEATURES]
        .to_numpy()
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            "Engineered features contain "
            "infinite values."
        )

    if not df[
        "TX_DATETIME"
    ].is_monotonic_increasing:
        raise ValueError(
            "Transactions are not "
            "chronologically ordered."
        )

    fraud_rate_columns = [
        column
        for column in ENGINEERED_FEATURES
        if "TERMINAL_FRAUD_RATE_" in column
    ]

    for column in fraud_rate_columns:

        if not df[column].between(
            0,
            1,
        ).all():
            raise ValueError(
                f"{column} contains values "
                "outside [0, 1]."
            )

    print(
        "Engineered feature validation passed."
    )


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the feature engineering pipeline."""

    print(
        "Building transaction features..."
    )

    df = add_transaction_features(
        df
    )

    print(
        "Building customer historical features..."
    )

    df = add_customer_features(
        df
    )

    print(
        "Building terminal historical features..."
    )

    df = add_terminal_features(
        df
    )

    print(
        "Cleaning engineered features..."
    )

    df = clean_engineered_features(
        df
    )

    # Restore chronological transaction order.
    df = df.sort_values(
        [
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    ).reset_index(
        drop=True
    )

    return df


def main() -> None:
    """Build and save the engineered feature dataset."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            "Raw dataset not found. "
            "Run `python -m src.data.download` "
            "first."
        )

    print(
        "Loading raw transactions..."
    )

    transactions = pd.read_parquet(
        RAW_DATA_FILE
    )

    print(
        f"Loaded "
        f"{len(transactions):,} "
        f"transactions."
    )

    features = build_features(
        transactions
    )

    validate_engineered_features(
        features
    )

    FEATURE_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        FEATURE_DATA_FILE,
        index=False,
    )

    print()

    print(
        "Feature engineering completed."
    )

    print(
        f"Rows: "
        f"{len(features):,}"
    )

    print(
        f"Columns: "
        f"{features.shape[1]}"
    )

    print(
        f"Saved to: "
        f"{FEATURE_DATA_FILE}"
    )


if __name__ == "__main__":
    main()