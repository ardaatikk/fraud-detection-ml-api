import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.paths import (
    BEST_MODEL_PATH,
    FIGURE_DIR,
    FINAL_TEST_RESULTS_PATH,
    TEST_DATA_FILE,
    THRESHOLD_PATH,
)
from src.config.settings import (
    TARGET_COLUMN,
)


def save_evaluation_figures(
    y_true: pd.Series,
    predictions,
    probabilities,
    pr_auc: float,
    roc_auc: float,
) -> None:
    """Create and save final evaluation figures."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Confusion matrix
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=[
            "Legitimate",
            "Fraud",
        ],
        values_format=",d",
        ax=ax,
    )

    ax.set_title(
        "Fraud Detection Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "confusion_matrix.png",
        dpi=150,
    )

    plt.close(fig)

    # Precision-Recall curve
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name=f"PR-AUC = {pr_auc:.3f}",
        ax=ax,
    )

    ax.set_title(
        "Precision-Recall Curve"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "precision_recall_curve.png",
        dpi=150,
    )

    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name=f"ROC-AUC = {roc_auc:.3f}",
        ax=ax,
    )

    ax.set_title(
        "ROC Curve"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "roc_curve.png",
        dpi=150,
    )

    plt.close(fig)


def main() -> None:
    """
    Evaluate the selected model once on the
    held-out test set.
    """

    print(
        "Loading final model, threshold "
        "and test set..."
    )

    test = pd.read_parquet(
        TEST_DATA_FILE
    )

    model_artifact = joblib.load(
        BEST_MODEL_PATH
    )

    threshold_artifact = joblib.load(
        THRESHOLD_PATH
    )

    model = model_artifact[
        "model"
    ]

    feature_columns = model_artifact[
        "feature_columns"
    ]

    model_name = model_artifact[
        "model_name"
    ]

    threshold = threshold_artifact[
        "threshold"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        TARGET_COLUMN
    ]

    print(
        f"Model: {model_name}"
    )

    print(
        f"Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print(
        f"Fraud cases: "
        f"{int(y_test.sum()):,}"
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            predictions,
        ).ravel()
    )

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"PR-AUC:    "
        f"{pr_auc:.6f}"
    )

    print(
        f"ROC-AUC:   "
        f"{roc_auc:.6f}"
    )

    print(
        f"Precision: "
        f"{precision:.6f}"
    )

    print(
        f"Recall:    "
        f"{recall:.6f}"
    )

    print(
        f"F1:        "
        f"{f1:.6f}"
    )

    print()
    print("Confusion Matrix")
    print("-" * 60)

    print(f"TN: {tn:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TP: {tp:,}")

    print()
    print("Classification Report")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    results = pd.DataFrame(
        [
            {
                "model": model_name,
                "threshold": threshold,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "test_rows": len(test),
                "fraud_cases": int(
                    y_test.sum()
                ),
            }
        ]
    )

    save_evaluation_figures(
        y_true=y_test,
        predictions=predictions,
        probabilities=probabilities,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
    )

    FINAL_TEST_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        FINAL_TEST_RESULTS_PATH,
        index=False,
    )

    print(
        "\nFinal test results saved to: "
        f"{FINAL_TEST_RESULTS_PATH}"
    )

    print(
        f"Evaluation figures saved to: "
        f"{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()