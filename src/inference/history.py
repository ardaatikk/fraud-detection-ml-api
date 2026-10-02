from pathlib import Path

import pandas as pd

from src.config.paths import RAW_DATA_FILE


HISTORY_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
]


class TransactionHistoryStore:
    """
    Provide read-only access to historical transactions used during
    inference-time feature engineering.

    The current implementation uses the local historical transaction
    dataset. The storage backend can later be replaced without changing
    the prediction or API layers.
    """

    def __init__(
        self,
        data_file: Path = RAW_DATA_FILE,
    ) -> None:
        self.data_file = data_file
        self._transactions = self._load_history()

    def _load_history(self) -> pd.DataFrame:
        """Load and prepare historical transaction data."""

        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Historical transaction data not found: "
                f"{self.data_file}"
            )

        transactions = pd.read_parquet(
            self.data_file,
            columns=HISTORY_COLUMNS,
        )

        transactions = transactions.sort_values(
            [
                "TX_DATETIME",
                "TRANSACTION_ID",
            ]
        ).reset_index(drop=True)

        return transactions

    def get_customer_history(
        self,
        customer_id: int,
        before: pd.Timestamp,
        days: int,
    ) -> pd.DataFrame:
        """
        Return customer transactions from the requested historical window.

        The current transaction is excluded by requiring timestamps to be
        strictly earlier than `before`.
        """

        start = before - pd.Timedelta(
            days=days
        )

        history = self._transactions[
            (self._transactions["CUSTOMER_ID"] == customer_id)
            & (self._transactions["TX_DATETIME"] >= start)
            & (self._transactions["TX_DATETIME"] < before)
        ]

        return history.copy()

    def get_terminal_history(
        self,
        terminal_id: int,
        before: pd.Timestamp,
        days: int,
    ) -> pd.DataFrame:
        """
        Return terminal transactions from the requested historical window.

        Only observations before the transaction being scored are returned.
        """

        start = before - pd.Timedelta(
            days=days
        )

        history = self._transactions[
            (self._transactions["TERMINAL_ID"] == terminal_id)
            & (self._transactions["TX_DATETIME"] >= start)
            & (self._transactions["TX_DATETIME"] < before)
        ]

        return history.copy()