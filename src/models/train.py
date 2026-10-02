import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.features import FEATURE_SETS
from src.config.paths import (
    BASELINE_MODEL_PATH,
    BASELINE_RESULTS_PATH,
    MODEL_DIR,
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
)
from src.config.settings import (
    RANDOM_STATE,
    TARGET_COLUMN,
)


def build_model() -> Pipeline:
    """Create the Logistic Regression baseline pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities,
) -> dict[str, float]:
    """Calculate threshold-independent metrics."""

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }


def train_experiment(
    name: str,
    feature_columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    Pipeline,
    dict[str, float],
]:
    """Train and evaluate one feature-set experiment."""

    print()
    print("=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    model = build_model()

    print(
        "Training Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    metrics = evaluate_probabilities(
        y_validation,
        probabilities,
    )

    print(
        f"PR-AUC:  "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    return model, metrics


def main() -> None:
    """
    Run Logistic Regression feature-set
    ablation experiments.
    """

    print(
        "Loading train and validation splits..."
    )

    train = pd.read_parquet(
        TRAIN_DATA_FILE
    )

    validation = pd.read_parquet(
        VALIDATION_DATA_FILE
    )

    print(
        f"Train: "
        f"{len(train):,}"
    )

    print(
        f"Validation: "
        f"{len(validation):,}"
    )

    results = []

    trained_models = {}

    for (
        name,
        feature_columns,
    ) in FEATURE_SETS.items():

        model, metrics = train_experiment(
            name=name,
            feature_columns=feature_columns,
            train=train,
            validation=validation,
        )

        trained_models[name] = model

        results.append(
            {
                "experiment": name,
                "features": len(
                    feature_columns
                ),
                **metrics,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "pr_auc",
        ascending=False,
    )

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    best_experiment = (
        results_df
        .iloc[0]["experiment"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": (
                trained_models[
                    best_experiment
                ]
            ),
            "feature_columns": (
                FEATURE_SETS[
                    best_experiment
                ]
            ),
            "experiment": (
                best_experiment
            ),
        },
        BASELINE_MODEL_PATH,
    )

    results_df.to_csv(
        BASELINE_RESULTS_PATH,
        index=False,
    )

    print()

    print(
        f"Best validation experiment: "
        f"{best_experiment}"
    )

    print(
        f"Model saved to: "
        f"{BASELINE_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()