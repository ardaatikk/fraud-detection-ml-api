import numpy as np
import pandas as pd
import pytest

from src.models.threshold import (
    evaluate_threshold,
    find_best_f1_threshold,
)


def test_find_best_f1_threshold() -> None:
    """Best threshold should maximize F1 for known probabilities."""

    y_true = pd.Series(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    threshold, f1 = find_best_f1_threshold(
        y_true,
        probabilities,
    )

    assert threshold == pytest.approx(0.80)
    assert f1 == pytest.approx(1.0)


def test_evaluate_threshold_perfect_classification() -> None:
    """A separating threshold should produce perfect metrics."""

    y_true = pd.Series(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    metrics = evaluate_threshold(
        y_true,
        probabilities,
        threshold=0.50,
    )

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_evaluate_threshold_confusion_matrix() -> None:
    """Threshold evaluation should return correct confusion counts."""

    y_true = pd.Series(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.70, 0.40, 0.90]
    )

    metrics = evaluate_threshold(
        y_true,
        probabilities,
        threshold=0.50,
    )

    # Predictions:
    # 0.10 -> 0 -> TN
    # 0.70 -> 1 -> FP
    # 0.40 -> 0 -> FN
    # 0.90 -> 1 -> TP

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_threshold_boundary_is_inclusive() -> None:
    """Probability equal to threshold should be classified positive."""

    y_true = pd.Series(
        [0, 1]
    )

    probabilities = np.array(
        [0.49, 0.50]
    )

    metrics = evaluate_threshold(
        y_true,
        probabilities,
        threshold=0.50,
    )

    assert metrics["tn"] == 1
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0