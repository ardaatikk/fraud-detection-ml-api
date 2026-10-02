import pickle
import sys
import types
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config.paths import (
    RAW_DATA_DIR,
    RAW_DATA_FILE,
)


# ------------------------------------------------------------------
# Dataset source
# ------------------------------------------------------------------

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "Fraud-Detection-Handbook/"
    "simulated-data-raw/main/data"
)

START_DATE = "2018-04-01"
END_DATE = "2018-09-30"


# ------------------------------------------------------------------
# Pandas pickle compatibility
# ------------------------------------------------------------------

def configure_pickle_compatibility() -> None:
    """
    Configure compatibility for dataset files serialized
    with older pandas versions.

    The source pickle files reference index classes that
    were removed from newer pandas releases.
    """

    numeric_module = types.ModuleType(
        "pandas.core.indexes.numeric"
    )

    numeric_module.Int64Index = pd.Index
    numeric_module.UInt64Index = pd.Index
    numeric_module.Float64Index = pd.Index

    sys.modules[
        "pandas.core.indexes.numeric"
    ] = numeric_module


# ------------------------------------------------------------------
# Download utilities
# ------------------------------------------------------------------

def download_file(
    url: str,
    destination: Path,
) -> None:
    """Download a file to the given destination."""

    response = requests.get(
        url,
        stream=True,
        timeout=60,
    )

    response.raise_for_status()

    with destination.open(
        "wb"
    ) as file:

        for chunk in response.iter_content(
            chunk_size=8192
        ):
            if chunk:
                file.write(
                    chunk
                )


def download_dataset() -> None:
    """
    Download and combine the Fraud Detection
    Handbook simulated transaction dataset.
    """

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if RAW_DATA_FILE.exists():

        print(
            f"Dataset already exists: "
            f"{RAW_DATA_FILE}"
        )

        print(
            "Skipping download."
        )

        return

    configure_pickle_compatibility()

    dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    daily_frames = []

    print(
        f"Downloading "
        f"{len(dates)} "
        f"daily transaction files..."
    )

    for date in tqdm(
        dates,
        desc="Downloading",
    ):

        date_string = date.strftime(
            "%Y-%m-%d"
        )

        filename = (
            f"{date_string}.pkl"
        )

        url = (
            f"{BASE_URL}/{filename}"
        )

        temporary_file = (
            RAW_DATA_DIR
            / filename
        )

        try:
            download_file(
                url,
                temporary_file,
            )

            with temporary_file.open(
                "rb"
            ) as file:

                daily_data = pickle.load(
                    file
                )

            daily_frames.append(
                daily_data
            )

        finally:
            if temporary_file.exists():
                temporary_file.unlink()

    print(
        "Combining daily transaction files..."
    )

    transactions = pd.concat(
        daily_frames,
        ignore_index=True,
    )

    transactions.to_parquet(
        RAW_DATA_FILE,
        index=False,
    )

    fraud_count = int(
        transactions[
            "TX_FRAUD"
        ].sum()
    )

    fraud_rate = (
        transactions[
            "TX_FRAUD"
        ].mean()
        * 100
    )

    print()
    print(
        "Dataset downloaded successfully."
    )

    print(
        f"Path: "
        f"{RAW_DATA_FILE}"
    )

    print(
        f"Transactions: "
        f"{len(transactions):,}"
    )

    print(
        f"Fraud transactions: "
        f"{fraud_count:,}"
    )

    print(
        f"Fraud rate: "
        f"{fraud_rate:.4f}%"
    )

    print()
    print("Columns:")

    for column in transactions.columns:
        print(
            f"  - {column}"
        )


if __name__ == "__main__":
    download_dataset()