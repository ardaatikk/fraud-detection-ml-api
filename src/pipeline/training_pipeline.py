import pandas as pd

from src.config.paths import (
    FEATURE_DATA_FILE,
    RAW_DATA_FILE,
)
from src.data.download import download_dataset
from src.data.split import main as create_splits
from src.data.validate import validate_dataset
from src.features.build_features import (
    main as build_feature_dataset,
    validate_engineered_features,
)
from src.models.compare import main as compare_models
from src.models.evaluate import main as evaluate_model
from src.models.threshold import main as select_threshold
from src.models.train import main as train_baseline


def run_training_pipeline() -> None:
    """Run the complete fraud detection training pipeline."""

    print("=" * 60)
    print("FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Dataset
    # ------------------------------------------------------------------

    print("\n[1/8] Checking dataset...")

    if not RAW_DATA_FILE.exists():
        print(
            "Dataset not found. "
            "Starting download..."
        )
        download_dataset()
    else:
        print(
            "Dataset already exists. "
            "Skipping download."
        )

    # ------------------------------------------------------------------
    # Step 2: Raw data validation
    # ------------------------------------------------------------------

    print("\n[2/8] Validating raw dataset...")

    transactions = pd.read_parquet(
        RAW_DATA_FILE
    )

    print(
        f"Loaded {len(transactions):,} transactions."
    )

    validate_dataset(
        transactions
    )

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------

    print("\n[3/8] Building features...")

    if not FEATURE_DATA_FILE.exists():
        build_feature_dataset()
    else:
        print(
            "Feature dataset already exists. "
            "Skipping feature engineering."
        )

        features = pd.read_parquet(
            FEATURE_DATA_FILE
        )

        validate_engineered_features(
            features
        )

    # ------------------------------------------------------------------
    # Step 4: Train / validation / test split
    # ------------------------------------------------------------------

    print("\n[4/8] Creating chronological splits...")

    create_splits()

    # ------------------------------------------------------------------
    # Step 5: Baseline experiments
    # ------------------------------------------------------------------

    print("\n[5/8] Running baseline experiments...")

    train_baseline()

    # ------------------------------------------------------------------
    # Step 6: Model comparison
    # ------------------------------------------------------------------

    print("\n[6/8] Comparing candidate models...")

    compare_models()

    # ------------------------------------------------------------------
    # Step 7: Threshold selection
    # ------------------------------------------------------------------

    print("\n[7/8] Selecting decision threshold...")

    select_threshold()

    # ------------------------------------------------------------------
    # Step 8: Final test evaluation
    # ------------------------------------------------------------------

    print("\n[8/8] Evaluating final model...")

    evaluate_model()

    print()
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()