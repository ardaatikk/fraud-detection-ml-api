from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.features import (
    FEATURE_SETS,
    MODEL_FEATURES,
)
from src.models.compare import build_models
from src.models.train import build_model


def test_model_features_are_unique() -> None:
    """Model feature list should not contain duplicates."""

    assert len(MODEL_FEATURES) == len(
        set(MODEL_FEATURES)
    )


def test_model_uses_expected_number_of_features() -> None:
    """Final model configuration should contain 19 features."""

    assert len(MODEL_FEATURES) == 19


def test_all_engineered_feature_set_matches_model_features() -> None:
    """Full experiment feature set should match production features."""

    assert (
        FEATURE_SETS["all_engineered"]
        == MODEL_FEATURES
    )


def test_baseline_model_is_pipeline() -> None:
    """Baseline model should use scaling and Logistic Regression."""

    model = build_model()

    assert isinstance(
        model,
        Pipeline,
    )

    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps

    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )


def test_baseline_logistic_regression_is_class_balanced() -> None:
    """Baseline should compensate for the imbalanced target."""

    model = build_model()

    classifier = model.named_steps[
        "classifier"
    ]

    assert classifier.class_weight == "balanced"


def test_candidate_models_are_available() -> None:
    """Model comparison should expose both candidate models."""

    models = build_models()

    assert set(models.keys()) == {
        "logistic_regression",
        "hist_gradient_boosting",
    }


def test_comparison_logistic_regression_is_pipeline() -> None:
    """Comparison Logistic Regression should retain preprocessing."""

    models = build_models()

    assert isinstance(
        models["logistic_regression"],
        Pipeline,
    )


def test_hist_gradient_boosting_configuration() -> None:
    """Gradient boosting candidate should use expected configuration."""

    models = build_models()

    model = models[
        "hist_gradient_boosting"
    ]

    assert isinstance(
        model,
        HistGradientBoostingClassifier,
    )

    assert model.learning_rate == 0.1
    assert model.max_iter == 200
    assert model.max_leaf_nodes == 31
    assert model.l2_regularization == 1.0
    assert model.random_state == 42