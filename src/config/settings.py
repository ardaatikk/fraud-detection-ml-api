# ------------------------------------------------------------------
# Dataset split configuration
# ------------------------------------------------------------------

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

RANDOM_STATE = 42


# ------------------------------------------------------------------
# Target
# ------------------------------------------------------------------

TARGET_COLUMN = "TX_FRAUD"


# ------------------------------------------------------------------
# Configuration validation
# ------------------------------------------------------------------

SPLIT_RATIO_TOTAL = (
    TRAIN_RATIO
    + VALIDATION_RATIO
    + TEST_RATIO
)

if not abs(
    SPLIT_RATIO_TOTAL - 1.0
) < 1e-9:
    raise ValueError(
        "Dataset split ratios must sum to 1.0. "
        f"Current total: {SPLIT_RATIO_TOTAL}"
    )