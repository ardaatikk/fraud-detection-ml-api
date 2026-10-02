import pandas as pd

from src.data.split import time_based_split


def make_split_dataset(
    n_rows: int = 100,
) -> pd.DataFrame:
    """Create chronological synthetic data for split tests."""

    return pd.DataFrame(
        {
            "TRANSACTION_ID": range(n_rows),
            "TX_DATETIME": pd.date_range(
                start="2018-01-01",
                periods=n_rows,
                freq="h",
            ),
            "TX_FRAUD": [
                index % 10 == 0
                for index in range(n_rows)
            ],
        }
    )


def test_split_preserves_all_rows() -> None:
    """Every input row should appear in exactly one split."""

    df = make_split_dataset()

    train, validation, test = time_based_split(df)

    assert (
        len(train)
        + len(validation)
        + len(test)
        == len(df)
    )

    combined_ids = pd.concat(
        [
            train["TRANSACTION_ID"],
            validation["TRANSACTION_ID"],
            test["TRANSACTION_ID"],
        ]
    )

    assert combined_ids.nunique() == len(df)


def test_split_is_chronologically_separated() -> None:
    """Train, validation and test must be temporally separated."""

    df = make_split_dataset()

    train, validation, test = time_based_split(df)

    assert (
        train["TX_DATETIME"].max()
        < validation["TX_DATETIME"].min()
    )

    assert (
        validation["TX_DATETIME"].max()
        < test["TX_DATETIME"].min()
    )


def test_split_is_chronologically_sorted() -> None:
    """Each resulting split should remain chronological."""

    df = (
        make_split_dataset()
        .sample(
            frac=1,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    train, validation, test = time_based_split(df)

    assert train["TX_DATETIME"].is_monotonic_increasing
    assert validation["TX_DATETIME"].is_monotonic_increasing
    assert test["TX_DATETIME"].is_monotonic_increasing


def test_split_ratios_are_approximately_correct() -> None:
    """Unique timestamps should produce approximately 70/15/15 splits."""

    df = make_split_dataset(
        n_rows=1000
    )

    train, validation, test = time_based_split(df)

    assert abs(
        len(train) / len(df) - 0.70
    ) < 0.01

    assert abs(
        len(validation) / len(df) - 0.15
    ) < 0.01

    assert abs(
        len(test) / len(df) - 0.15
    ) < 0.01


def test_identical_timestamps_are_not_split() -> None:
    """Transactions with identical timestamps must stay together."""

    df = make_split_dataset(
        n_rows=100
    )

    # Put several transactions exactly around the expected
    # train boundary on the same timestamp.
    shared_timestamp = pd.Timestamp(
        "2018-01-03 22:00:00"
    )

    df.loc[
        68:72,
        "TX_DATETIME",
    ] = shared_timestamp

    train, validation, test = time_based_split(df)

    locations = []

    for split_name, split_df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if (
            split_df["TX_DATETIME"]
            == shared_timestamp
        ).any():
            locations.append(split_name)

    assert len(locations) == 1