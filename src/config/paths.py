from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ------------------------------------------------------------------
# Data directories
# ------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SPLIT_DIR = PROCESSED_DATA_DIR / "splits"


# ------------------------------------------------------------------
# Dataset files
# ------------------------------------------------------------------

RAW_DATA_FILE = (
    RAW_DATA_DIR
    / "transactions.parquet"
)

FEATURE_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "features.parquet"
)

TRAIN_DATA_FILE = (
    SPLIT_DIR
    / "train.parquet"
)

VALIDATION_DATA_FILE = (
    SPLIT_DIR
    / "validation.parquet"
)

TEST_DATA_FILE = (
    SPLIT_DIR
    / "test.parquet"
)


# ------------------------------------------------------------------
# Artifact directories
# ------------------------------------------------------------------

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR = ARTIFACT_DIR / "models"

FIGURE_DIR = ARTIFACT_DIR / "figures"


# ------------------------------------------------------------------
# Model artifacts
# ------------------------------------------------------------------

BASELINE_MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression_baseline.joblib"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_validation_model.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR
    / "threshold.joblib"
)


# ------------------------------------------------------------------
# Model results
# ------------------------------------------------------------------

BASELINE_RESULTS_PATH = (
    MODEL_DIR
    / "baseline_results.csv"
)

MODEL_COMPARISON_PATH = (
    MODEL_DIR
    / "model_comparison.csv"
)

THRESHOLD_COMPARISON_PATH = (
    MODEL_DIR
    / "threshold_comparison.csv"
)

FINAL_TEST_RESULTS_PATH = (
    MODEL_DIR
    / "final_test_results.csv"
)