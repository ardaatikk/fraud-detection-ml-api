TRANSACTION_FEATURES = [
    "TX_AMOUNT",
]

TEMPORAL_FEATURES = [
    "TX_HOUR",
    "TX_DAY_OF_WEEK",
]

CUSTOMER_FEATURES = [
    "CUSTOMER_TX_COUNT_1D",
    "CUSTOMER_TX_COUNT_7D",
    "CUSTOMER_TX_COUNT_30D",
    "CUSTOMER_AVG_AMOUNT_1D",
    "CUSTOMER_AVG_AMOUNT_7D",
    "CUSTOMER_AVG_AMOUNT_30D",
    "AMOUNT_TO_CUSTOMER_AVG_1D",
    "AMOUNT_TO_CUSTOMER_AVG_7D",
    "AMOUNT_TO_CUSTOMER_AVG_30D",
]

TERMINAL_FEATURES = [
    "TERMINAL_TX_COUNT_1D",
    "TERMINAL_TX_COUNT_7D",
    "TERMINAL_TX_COUNT_30D",
    "TERMINAL_FRAUD_COUNT_7D",
    "TERMINAL_FRAUD_COUNT_30D",
    "TERMINAL_FRAUD_RATE_7D",
    "TERMINAL_FRAUD_RATE_30D",
]


# Features created during feature engineering.
ENGINEERED_FEATURES = (
    TEMPORAL_FEATURES
    + CUSTOMER_FEATURES
    + TERMINAL_FEATURES
)


# Complete feature set used by the final model.
MODEL_FEATURES = (
    TRANSACTION_FEATURES
    + ENGINEERED_FEATURES
)


# Feature sets used for ablation experiments.
FEATURE_SETS = {
    "amount_only": (
        TRANSACTION_FEATURES
    ),

    "amount_temporal": (
        TRANSACTION_FEATURES
        + TEMPORAL_FEATURES
    ),

    "customer_behavior": (
        TRANSACTION_FEATURES
        + TEMPORAL_FEATURES
        + CUSTOMER_FEATURES
    ),

    "terminal_behavior": (
        TRANSACTION_FEATURES
        + TEMPORAL_FEATURES
        + TERMINAL_FEATURES
    ),

    "all_engineered": (
        MODEL_FEATURES
    ),
}