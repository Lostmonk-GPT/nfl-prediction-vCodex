import pandas as pd
import pytest

from nfl_pred.model.splits import time_series_splits


def _collect_weeks(df, indices):
    return sorted(df.loc[indices, "week"].unique())


def test_time_series_splits_forward_chaining():
    df = pd.DataFrame(
        {
            "week": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "team": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    folds = list(time_series_splits(df, n_splits=2, min_train_weeks=2))
    assert len(folds) == 2

    for i, (train_idx, val_idx) in enumerate(folds):
        train_weeks = _collect_weeks(df, train_idx)
        val_weeks = _collect_weeks(df, val_idx)

        assert len(val_weeks) == 1
        assert val_weeks[0] == 3 + i
        # Ensure all training weeks precede the validation week.
        assert max(train_weeks) < val_weeks[0]
        # Ensure training accumulates monotonically.
        if i > 0:
            prev_train_weeks = _collect_weeks(df, folds[i - 1][0])
            assert prev_train_weeks == train_weeks[: len(prev_train_weeks)]


def test_time_series_splits_allows_full_range_when_n_splits_none():
    df = pd.DataFrame({"week": [1, 1, 2, 3, 3, 4], "team": list("ABCDEF")})

    folds = list(time_series_splits(df, n_splits=None, min_train_weeks=2))
    # Weeks available for validation: 3 and 4 -> 2 folds.
    assert len(folds) == 2
    val_weeks = [_collect_weeks(df, val_idx)[0] for _, val_idx in folds]
    assert val_weeks == [3, 4]


def test_time_series_splits_raises_when_insufficient_history():
    df = pd.DataFrame({"week": [1, 1, 2, 2], "team": list("ABCD")})

    with pytest.raises(ValueError):
        list(time_series_splits(df, n_splits=1, min_train_weeks=3))
