from time import perf_counter

import joblib
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.features import (
    MODEL_FEATURES,
)
from src.config.paths import (
    BEST_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DIR,
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
)
from src.config.settings import (
    RANDOM_STATE,
    TARGET_COLUMN,
)


def build_models() -> dict:
    """Create candidate classification models."""

    return {
        "logistic_regression": Pipeline(
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
        ),

        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_model(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a model using threshold-independent
    classification metrics.
    """

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    return {
        "pr_auc": average_precision_score(
            y_validation,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
    }


def main() -> None:
    """
    Train candidate models and select the model
    with the highest validation PR-AUC.
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

    X_train = train[
        MODEL_FEATURES
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_validation = validation[
        MODEL_FEATURES
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    models = build_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        print()
        print("=" * 60)
        print(f"MODEL: {name}")
        print("=" * 60)

        start = perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            perf_counter()
            - start
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        print(
            f"Training time: "
            f"{training_seconds:.2f} seconds"
        )

        print(
            f"PR-AUC:  "
            f"{metrics['pr_auc']:.6f}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

        trained_models[
            name
        ] = model

        results.append(
            {
                "model": name,
                "pr_auc": (
                    metrics["pr_auc"]
                ),
                "roc_auc": (
                    metrics["roc_auc"]
                ),
                "training_seconds": (
                    training_seconds
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    best_model_name = (
        results_df
        .iloc[0]["model"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": (
                trained_models[
                    best_model_name
                ]
            ),
            "feature_columns": (
                MODEL_FEATURES
            ),
            "model_name": (
                best_model_name
            ),
        },
        BEST_MODEL_PATH,
    )

    results_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    print()

    print(
        f"Best validation model: "
        f"{best_model_name}"
    )

    print(
        f"Model saved to: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Comparison results saved to: "
        f"{MODEL_COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()