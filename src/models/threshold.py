import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config.paths import (
    BEST_MODEL_PATH,
    MODEL_DIR,
    THRESHOLD_COMPARISON_PATH,
    THRESHOLD_PATH,
    VALIDATION_DATA_FILE,
)
from src.config.settings import (
    TARGET_COLUMN,
)


def find_best_f1_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """
    Find the probability threshold that maximizes
    the F1 score on the validation set.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    # precision and recall contain one extra value
    # compared with thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-12
        )
    )

    best_index = np.argmax(
        f1_scores
    )

    return (
        float(
            thresholds[
                best_index
            ]
        ),
        float(
            f1_scores[
                best_index
            ]
        ),
    )


def evaluate_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """
    Evaluate classification performance at
    a specific probability threshold.
    """

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
        ).ravel()
    )

    return {
        "threshold": threshold,

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    """
    Select the classification threshold using
    validation-set F1 score.
    """

    print(
        "Loading validation data and model..."
    )

    validation = pd.read_parquet(
        VALIDATION_DATA_FILE
    )

    artifact = joblib.load(
        BEST_MODEL_PATH
    )

    model = artifact[
        "model"
    ]

    feature_columns = artifact[
        "feature_columns"
    ]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    (
        best_threshold,
        best_f1,
    ) = find_best_f1_threshold(
        y_validation,
        probabilities,
    )

    print(
        f"\nBest F1 threshold: "
        f"{best_threshold:.6f}"
    )

    print(
        f"Validation F1: "
        f"{best_f1:.6f}"
    )

    print()
    print("=" * 60)
    print("THRESHOLD COMPARISON")
    print("=" * 60)

    thresholds_to_test = sorted(
        {
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            best_threshold,
        }
    )

    results = []

    for threshold in thresholds_to_test:

        metrics = evaluate_threshold(
            y_validation,
            probabilities,
            threshold,
        )

        results.append(
            metrics
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "threshold": (
                best_threshold
            ),
            "selection_metric": (
                "f1"
            ),
            "validation_f1": (
                best_f1
            ),
        },
        THRESHOLD_PATH,
    )

    results_df.to_csv(
        THRESHOLD_COMPARISON_PATH,
        index=False,
    )

    print()

    print(
        f"Selected threshold saved to: "
        f"{THRESHOLD_PATH}"
    )

    print(
        f"Threshold comparison saved to: "
        f"{THRESHOLD_COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()