import pandas as pd

from src.config.features import MODEL_FEATURES
from src.inference.history import TransactionHistoryStore


class InferenceFeatureBuilder:
    """
    Build model features for a single incoming transaction using only
    information available before its timestamp.
    """

    def __init__(
        self,
        history_store: TransactionHistoryStore,
    ) -> None:
        self.history_store = history_store

    def build(
        self,
        *,
        customer_id: int,
        terminal_id: int,
        tx_amount: float,
        tx_datetime: pd.Timestamp,
    ) -> dict[str, float]:
        """Build the complete production feature vector."""

        timestamp = pd.Timestamp(
            tx_datetime
        )

        features: dict[str, float] = {
            "TX_AMOUNT": float(tx_amount),
            "TX_HOUR": float(timestamp.hour),
            "TX_DAY_OF_WEEK": float(
                timestamp.dayofweek
            ),
        }

        self._add_customer_features(
            features=features,
            customer_id=customer_id,
            tx_amount=tx_amount,
            tx_datetime=timestamp,
        )

        self._add_terminal_features(
            features=features,
            terminal_id=terminal_id,
            tx_datetime=timestamp,
        )

        self._validate_features(
            features
        )

        return features

    def _add_customer_features(
        self,
        *,
        features: dict[str, float],
        customer_id: int,
        tx_amount: float,
        tx_datetime: pd.Timestamp,
    ) -> None:
        """Add historical customer behavior features."""

        for days in (
            1,
            7,
            30,
        ):
            history = (
                self.history_store
                .get_customer_history(
                    customer_id=customer_id,
                    before=tx_datetime,
                    days=days,
                )
            )

            count = len(history)

            if count > 0:
                average_amount = float(
                    history["TX_AMOUNT"].mean()
                )

                if average_amount != 0:
                    amount_ratio = (
                        float(tx_amount)
                        / average_amount
                    )
                else:
                    amount_ratio = 0.0

            else:
                average_amount = 0.0
                amount_ratio = 0.0

            suffix = f"{days}D"

            features[
                f"CUSTOMER_TX_COUNT_{suffix}"
            ] = float(count)

            features[
                f"CUSTOMER_AVG_AMOUNT_{suffix}"
            ] = average_amount

            features[
                f"AMOUNT_TO_CUSTOMER_AVG_{suffix}"
            ] = amount_ratio

    def _add_terminal_features(
        self,
        *,
        features: dict[str, float],
        terminal_id: int,
        tx_datetime: pd.Timestamp,
    ) -> None:
        """Add historical terminal activity and fraud features."""

        histories = {}

        for days in (
            1,
            7,
            30,
        ):
            histories[days] = (
                self.history_store
                .get_terminal_history(
                    terminal_id=terminal_id,
                    before=tx_datetime,
                    days=days,
                )
            )

            features[
                f"TERMINAL_TX_COUNT_{days}D"
            ] = float(
                len(histories[days])
            )

        for days in (
            7,
            30,
        ):
            history = histories[days]

            if history.empty:
                fraud_count = 0.0
                fraud_rate = 0.0
            else:
                fraud_count = float(
                    history["TX_FRAUD"].sum()
                )

                fraud_rate = float(
                    history["TX_FRAUD"].mean()
                )

            features[
                f"TERMINAL_FRAUD_COUNT_{days}D"
            ] = fraud_count

            features[
                f"TERMINAL_FRAUD_RATE_{days}D"
            ] = fraud_rate

    @staticmethod
    def _validate_features(
        features: dict[str, float],
    ) -> None:
        """Ensure inference features match the trained model contract."""

        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if feature not in features
        ]

        unexpected_features = [
            feature
            for feature in features
            if feature not in MODEL_FEATURES
        ]

        if missing_features:
            raise ValueError(
                "Missing inference features: "
                f"{missing_features}"
            )

        if unexpected_features:
            raise ValueError(
                "Unexpected inference features: "
                f"{unexpected_features}"
            )