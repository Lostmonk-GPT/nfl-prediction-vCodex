"""Tests for the baseline modeling wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_pred.model.baseline import BaselineClassifier


@pytest.fixture()
def toy_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """Return a small dataset with numeric and categorical features."""

    X = pd.DataFrame(
        {
            "yards_per_play": [4.1, 5.2, 3.8, 6.4, 5.9, 4.5],
            "opponent": ["NYJ", "NYJ", "BUF", "BUF", "MIA", "MIA"],
            "is_home": [True, False, True, False, True, False],
        }
    )
    y = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    return X, y


def test_baseline_classifier_predict_proba_shape_and_bounds(toy_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Model outputs probabilities with correct shape and numeric bounds."""

    X, y = toy_dataset
    clf = BaselineClassifier(random_state=0)
    clf.fit(X, y)

    # Provide columns in a different order to ensure alignment logic works.
    X_reordered = X[["opponent", "yards_per_play", "is_home"]]
    probs = clf.predict_proba(X_reordered)

    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(clf.classes_) == {0, 1}


def test_baseline_classifier_missing_columns_raises(toy_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Predicting without required columns should raise a descriptive error."""

    X, y = toy_dataset
    clf = BaselineClassifier(random_state=0)
    clf.fit(X, y)

    X_missing = X.drop(columns=["is_home"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        clf.predict_proba(X_missing)
