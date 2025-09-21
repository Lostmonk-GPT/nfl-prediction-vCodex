"""Rolling window feature utilities.

This module provides helpers to compute rolling aggregates (means and rates)
for team/week level feature engineering tasks.

Example
-------
>>> import pandas as pd
>>> from nfl_pred.features.windows import RollingMetric, compute_group_rolling_windows
>>> df = pd.DataFrame(
...     {
...         "team": ["A", "A", "A", "B", "B"],
...         "week": [1, 2, 3, 1, 2],
...         "yards": [100, 110, 90, 80, 120],
...         "successes": [5, 6, 4, 3, 5],
...         "plays": [10, 12, 9, 8, 11],
...     }
... )
>>> metrics = [
...     RollingMetric(name="yards", value_column="yards", statistic="mean"),
...     RollingMetric(
...         name="success_rate",
...         value_column="successes",
...         denominator_column="plays",
...         statistic="rate",
...     ),
... ]
>>> windows = {"w2": 2, "season": None}
>>> compute_group_rolling_windows(
...     df,
...     metrics=metrics,
...     group_keys=["team"],
...     order_key="week",
...     window_lengths=windows,
... )
   team  week  yards_w2  yards_season  success_rate_w2  success_rate_season
0    A     1     100.0          100.0               0.5                  0.5
1    A     2     105.0          105.0               0.5                  0.5
2    A     3     100.0          100.0               0.5                  0.5
3    B     1      80.0           80.0               0.375                0.375
4    B     2     100.0          100.0               0.444444             0.444444
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, MutableMapping, Sequence

import pandas as pd

Statistic = Literal["mean", "sum", "rate"]


@dataclass(frozen=True)
class RollingMetric:
    """Configuration for a rolling aggregate.

    Parameters
    ----------
    name:
        Base name for the output columns. Window suffixes will be appended to
        this value using an underscore separator.
    value_column:
        Column containing the values to aggregate.
    statistic:
        Type of aggregate to compute. Supported values: ``"mean"``, ``"sum"``,
        and ``"rate"``.
    denominator_column:
        Required when ``statistic`` is ``"rate"``. Represents the denominator
        for rate calculations (e.g., plays when computing success rate).
    min_periods:
        Minimum number of observations required to produce a value. Defaults to
        ``1`` to allow partial windows at the start of a season.
    """

    name: str
    value_column: str
    statistic: Statistic
    denominator_column: str | None = None
    min_periods: int = 1

    def __post_init__(self) -> None:  # noqa: D401 - validation helper
        """Validate the metric configuration."""
        if self.statistic not in {"mean", "sum", "rate"}:
            raise ValueError(
                "statistic must be one of {'mean', 'sum', 'rate'}; "
                f"got {self.statistic!r}",
            )
        if self.statistic == "rate" and not self.denominator_column:
            raise ValueError("denominator_column is required when statistic='rate'")
        if self.min_periods < 1:
            raise ValueError("min_periods must be at least 1")


def _ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {missing}")


def _rolling_series(
    series: pd.Series,
    window: int | None,
    min_periods: int,
    agg: Literal["mean", "sum"],
) -> pd.Series:
    if window is None:
        rolling_obj = series.expanding(min_periods=min_periods)
    else:
        rolling_obj = series.rolling(window=window, min_periods=min_periods)
    if agg == "mean":
        return rolling_obj.mean()
    return rolling_obj.sum()


def compute_group_rolling_windows(
    df: pd.DataFrame,
    *,
    metrics: Sequence[RollingMetric],
    group_keys: Sequence[str],
    order_key: str,
    window_lengths: Mapping[str, int | None],
    asof_ts: pd.Timestamp | None = None,
    asof_column: str = "asof_ts",
) -> pd.DataFrame:
    """Compute rolling aggregates grouped by team (or other keys).

    Parameters
    ----------
    df:
        Input DataFrame containing the source metrics.
    metrics:
        Sequence of :class:`RollingMetric` definitions describing which
        aggregates to compute.
    group_keys:
        Columns used to group observations (e.g., ``["season", "team"]``).
    order_key:
        Column defining the temporal order inside each group (e.g., ``"week"``
        or ``"game_date"``).
    window_lengths:
        Mapping of suffix -> window length. Use ``None`` for season-to-date
        (expanding) windows. Example: ``{"w4": 4, "w8": 8, "season": None}``.
    asof_ts:
        Optional cutoff timestamp. When provided, rows with ``asof_column``
        values greater than ``asof_ts`` are excluded.
    asof_column:
        Column containing the timestamps for the ``asof_ts`` filter. Defaults to
        ``"asof_ts"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing ``group_keys``, ``order_key``, and one column per
        metric/window combination. The result is sorted by ``group_keys`` and
        ``order_key``.
    """

    if not group_keys:
        raise ValueError("group_keys must contain at least one column")
    if not window_lengths:
        raise ValueError("window_lengths cannot be empty")

    required_columns: set[str] = set(group_keys) | {order_key}
    for metric in metrics:
        required_columns.add(metric.value_column)
        if metric.denominator_column:
            required_columns.add(metric.denominator_column)
    if asof_ts is not None:
        required_columns.add(asof_column)

    _ensure_columns_exist(df, required_columns)

    working_df = df.copy()
    if asof_ts is not None:
        working_df = working_df.loc[working_df[asof_column] <= asof_ts].copy()

    sort_keys = [*group_keys, order_key]
    working_df = working_df.sort_values(sort_keys).reset_index(drop=True)
    grouped = working_df.groupby(list(group_keys), sort=False, group_keys=False)

    result: MutableMapping[str, pd.Series] = {
        key: working_df[key] for key in sort_keys
    }

    for metric in metrics:
        value_series = grouped[metric.value_column]
        denom_series = (
            grouped[metric.denominator_column]
            if metric.denominator_column is not None
            else None
        )

        for suffix, window in window_lengths.items():
            column_name = f"{metric.name}_{suffix}"
            if metric.statistic == "rate":
                assert denom_series is not None  # for type checkers
                num_values = value_series.apply(
                    lambda s: _rolling_series(s, window, metric.min_periods, "sum"),
                )
                den_values = denom_series.apply(
                    lambda s: _rolling_series(s, window, metric.min_periods, "sum"),
                )
                result[column_name] = num_values.divide(den_values).reset_index(
                    drop=True
                )
            else:
                agg_type: Literal["mean", "sum"] = (
                    "mean" if metric.statistic == "mean" else "sum"
                )
                result[column_name] = value_series.apply(
                    lambda s: _rolling_series(s, window, metric.min_periods, agg_type),
                ).reset_index(drop=True)

    return pd.DataFrame(result)
