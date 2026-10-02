from functools import lru_cache

import joblib
import pandas as pd

from src.config.features import MODEL_FEATURES
from src.config.paths import (
    BEST_MODEL_PATH,
    THRESHOLD_PATH,
)


class FraudPredictor:
    """Load trained artifacts and perform fraud predictions."""

    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load the trained model and selected decision threshold."""

        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {BEST_MODEL_PATH}"
            )

        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                f"Threshold artifact not found: {THRESHOLD_PATH}"
            )

        model_artifact = joblib.load(
            BEST_MODEL_PATH
        )

        threshold_artifact = joblib.load(
            THRESHOLD_PATH
        )

        self.model = model_artifact["model"]

        self.feature_columns = model_artifact[
            "feature_columns"
        ]

        self.model_name = model_artifact[
            "model_name"
        ]

        self.threshold = float(
            threshold_artifact["threshold"]
        )

        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        """Validate compatibility of deployed model artifacts."""

        if self.feature_columns != MODEL_FEATURES:
            raise ValueError(
                "Model artifact feature columns do not match "
                "the configured production features."
            )

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                "Decision threshold must be between 0 and 1."
            )

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            raise TypeError(
                "Loaded model does not support predict_proba()."
            )

    def predict(
        self,
        features: dict[str, float],
    ) -> dict:
        """Predict fraud probability for one engineered transaction."""

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in features
        ]

        if missing_features:
            raise ValueError(
                "Missing model features: "
                f"{missing_features}"
            )

        feature_frame = pd.DataFrame(
            [
                {
                    feature: features[feature]
                    for feature in self.feature_columns
                }
            ]
        )

        probability = float(
            self.model.predict_proba(
                feature_frame
            )[0, 1]
        )

        is_fraud = (
            probability >= self.threshold
        )

        return {
            "fraud_probability": probability,
            "is_fraud": bool(is_fraud),
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """
    Return a cached predictor instance.

    Model artifacts are loaded once per application process rather than
    once for every prediction request.
    """

    return FraudPredictor()