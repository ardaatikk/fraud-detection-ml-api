import pandas as pd

from src.config.paths import (
    FEATURE_DATA_FILE,
    SPLIT_DIR,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
)

from src.config.settings import (
    TARGET_COLUMN,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


def time_based_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split transactions chronologically into
    train, validation and test sets.

    Transactions sharing the same timestamp are
    kept within the same split to preserve strict
    temporal separation.
    """

    df = df.sort_values(
        [
            "TX_DATETIME",
            "TRANSACTION_ID",
        ]
    ).reset_index(
        drop=True
    )

    n_rows = len(df)

    train_position = int(
        n_rows * TRAIN_RATIO
    )

    validation_position = int(
        n_rows
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_cutoff = df.iloc[
        train_position
    ]["TX_DATETIME"]

    validation_cutoff = df.iloc[
        validation_position
    ]["TX_DATETIME"]

    train = df[
        df["TX_DATETIME"]
        < train_cutoff
    ].copy()

    validation = df[
        (
            df["TX_DATETIME"]
            >= train_cutoff
        )
        & (
            df["TX_DATETIME"]
            < validation_cutoff
        )
    ].copy()

    test = df[
        df["TX_DATETIME"]
        >= validation_cutoff
    ].copy()

    return (
        train,
        validation,
        test,
    )


def print_split_summary(
    name: str,
    df: pd.DataFrame,
) -> None:
    """Print summary information for a dataset split."""

    fraud_count = int(
        df[TARGET_COLUMN].sum()
    )

    fraud_rate = (
        df[TARGET_COLUMN].mean()
        * 100
    )

    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['TX_DATETIME'].min()} "
        f"→ "
        f"{df['TX_DATETIME'].max()}"
    )

    print(
        f"Frauds: {fraud_count:,}"
    )

    print(
        f"Fraud rate: "
        f"{fraud_rate:.4f}%"
    )


def main() -> None:
    """
    Create chronological train, validation
    and test datasets.
    """

    if not FEATURE_DATA_FILE.exists():
        raise FileNotFoundError(
            "Feature dataset not found. "
            "Run "
            "`python -m src.features.build_features` "
            "first."
        )

    print(
        "Loading engineered features..."
    )

    features = pd.read_parquet(
        FEATURE_DATA_FILE
    )

    print(
        f"Loaded "
        f"{len(features):,} "
        f"transactions."
    )

    train, validation, test = (
        time_based_split(
            features
        )
    )

    print_split_summary(
        "TRAIN",
        train,
    )

    print_split_summary(
        "VALIDATION",
        validation,
    )

    print_split_summary(
        "TEST",
        test,
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        TRAIN_DATA_FILE,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_DATA_FILE,
        index=False,
    )

    test.to_parquet(
        TEST_DATA_FILE,
        index=False,
    )

    print(
        "\nSplits saved successfully."
    )


if __name__ == "__main__":
    main()