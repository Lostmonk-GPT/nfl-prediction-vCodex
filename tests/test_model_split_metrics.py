"""Unit tests for time-series splits and basic metrics calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_pred.model.splits import time_series_splits
from nfl_pred.reporting.metrics import compute_classification_metrics


class TestTimeSeriesSplits:
    """Validate forward-chaining time-series split generation."""

    def test_splits_are_monotonic_and_cover_expected_weeks(self) -> None:
        """Each fold should train on earlier weeks and validate on the next week."""

        df = pd.DataFrame(
            {
                "week": np.repeat(np.arange(1, 6), 2),
                "team": ["A", "B"] * 5,
            }
        )

        splits = list(time_series_splits(df, n_splits=3, min_train_weeks=2))

        assert len(splits) == 3

        for fold_index, (train_idx, val_idx) in enumerate(splits):
            train_weeks = df.loc[train_idx, "week"].unique().tolist()
            val_weeks = df.loc[val_idx, "week"].unique().tolist()

            assert len(val_weeks) == 1, "Validation fold should only cover one week"
            val_week = val_weeks[0]

            expected_val_week = fold_index + 3  # with min_train_weeks=2 and weeks starting at 1
            assert val_week == expected_val_week

            expected_train_weeks = list(range(1, val_week))
            assert train_weeks == expected_train_weeks

            # Ensure temporal ordering: all training weeks precede the validation week
            assert max(train_weeks) < val_week


class TestClassificationMetrics:
    """Verify Brier score and log-loss calculations on simple inputs."""

    def test_compute_classification_metrics_matches_hand_calculation(self) -> None:
        """Metrics should align with manual calculations for tiny arrays."""

        df = pd.DataFrame(
            {
                "prob": [0.8, 0.4, 0.1],
                "label": [1, 0, 0],
            }
        )

        metrics = compute_classification_metrics(
            df,
            probability_column="prob",
            label_column="label",
        )

        assert metrics.shape == (1, 3)

        row = metrics.iloc[0]
        assert row["n_observations"] == 3
        assert row["brier_score"] == pytest.approx(0.07)
        assert row["log_loss"] == pytest.approx(0.2797765635793423)
