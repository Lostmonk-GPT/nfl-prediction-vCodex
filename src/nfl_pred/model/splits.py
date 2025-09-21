"""Cross-validation splits for forward-chaining NFL modeling experiments."""

from __future__ import annotations

from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd


def _sorted_unique(values: Iterable) -> pd.Index:
    """Return a pandas ``Index`` of sorted unique values preserving dtype."""
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Group column contains null values, cannot build time-series splits.")
    # ``mergesort`` is stable and works with datetime/int week identifiers.
    return pd.Index(series.sort_values(kind="mergesort").unique())


def time_series_splits(
    df: pd.DataFrame,
    group_col: str = "week",
    n_splits: int | None = None,
    min_train_weeks: int = 4,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield forward-chaining train/validation indices grouped by week.

    Parameters
    ----------
    df:
        Feature table containing one or more rows per team-week.
    group_col:
        Column that identifies the temporal grouping (defaults to ``"week"``).
    n_splits:
        Number of validation folds to generate. If ``None`` (default) the
        function will create as many folds as possible while satisfying the
        ``min_train_weeks`` constraint.
    min_train_weeks:
        Minimum number of distinct weeks to include in each training sample.

    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        Two numpy arrays containing the integer index locations for the train
        and validation subsets, respectively.

    Notes
    -----
    The generator enforces strictly increasing week order so that the
    validation week contains only observations that occur after the training
    weeks. Each fold uses one consecutive week for validation, preventing
    information leakage from future games.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "week": [1, 1, 2, 2, 3, 3, 4, 4],
    ...     "team": ["A", "B", "A", "B", "A", "B", "A", "B"],
    ... })
    >>> splits = list(time_series_splits(df, n_splits=2, min_train_weeks=2))
    >>> [df.loc[val_idx, "week"].unique().item() for _, val_idx in splits]
    [3, 4]
    >>> [df.loc[train_idx, "week"].unique().tolist() for train_idx, _ in splits]
    [[1, 2], [1, 2, 3]]
    """

    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not present in DataFrame.")
    if min_train_weeks < 1:
        raise ValueError("min_train_weeks must be at least 1.")

    unique_weeks = _sorted_unique(df[group_col])
    total_weeks = len(unique_weeks)
    if total_weeks <= min_train_weeks:
        raise ValueError(
            "Not enough distinct weeks to satisfy the minimum training window."
        )

    max_possible_splits = total_weeks - min_train_weeks
    if n_splits is None:
        splits_to_yield = max_possible_splits
    elif n_splits <= 0:
        raise ValueError("n_splits must be positive when provided.")
    else:
        if n_splits > max_possible_splits:
            raise ValueError(
                "Requested number of splits exceeds available validation weeks."
            )
        splits_to_yield = n_splits

    for fold in range(splits_to_yield):
        train_weeks = unique_weeks[: min_train_weeks + fold]
        val_week = unique_weeks[min_train_weeks + fold]

        train_mask = df[group_col].isin(train_weeks)
        val_mask = df[group_col] == val_week

        train_idx = df.index[train_mask].to_numpy()
        val_idx = df.index[val_mask].to_numpy()

        if len(val_idx) == 0:
            raise ValueError(f"Validation week {val_week!r} has no rows.")

        yield train_idx, val_idx
