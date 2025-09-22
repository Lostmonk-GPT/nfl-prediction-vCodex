"""Population Stability Index (PSI) computation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

_EPSILON = 1e-6
_DEFAULT_BINS = 10


@dataclass(frozen=True)
class PSISummary:
    """Container describing PSI outcomes across a feature set."""

    feature_psi: pd.DataFrame
    threshold: float

    @property
    def breached_features(self) -> list[str]:
        """Return features whose PSI meets or exceeds the configured threshold."""

        breached = self.feature_psi.loc[
            self.feature_psi["psi"] >= self.threshold, "feature"
        ]
        return breached.tolist()

    @property
    def breach_count(self) -> int:
        """Number of features whose PSI meets or exceeds the configured threshold."""

        return len(self.breached_features)


def _validate_inputs(reference: pd.Series, current: pd.Series) -> None:
    if reference.empty:
        raise ValueError("Reference series must contain at least one value.")
    if current.empty:
        raise ValueError("Current series must contain at least one value.")


def _resolve_bin_edges(reference: pd.Series, bins: int) -> np.ndarray:
    non_null = reference.dropna()
    if non_null.empty:
        # fall back to single catch-all bin
        return np.array([-np.inf, np.inf], dtype="float64")

    quantiles = np.linspace(0.0, 1.0, num=bins + 1)
    edges = np.quantile(non_null.to_numpy(), quantiles, method="linear")
    edges = np.unique(edges)

    if edges.size <= 1:
        return np.array([-np.inf, np.inf], dtype="float64")

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype("float64")


def _bin_counts(series: pd.Series, edges: np.ndarray) -> tuple[np.ndarray, int]:
    non_null = series.dropna()
    if non_null.empty:
        counts = np.zeros(len(edges) - 1, dtype="int64")
    else:
        buckets = pd.cut(
            non_null,
            bins=edges,
            include_lowest=True,
            right=False,
            duplicates="drop",
        )
        counts = buckets.value_counts(sort=False).to_numpy()

        # In the degenerate case where pd.cut collapsed bins, align with edge count.
        if counts.size != len(edges) - 1:
            aligned = np.zeros(len(edges) - 1, dtype="int64")
            series_cats = buckets.cat.categories
            for idx, category in enumerate(series_cats):
                # locate index of category in the original bins
                left = category.left
                bin_idx = np.searchsorted(edges[:-1], left)
                aligned[bin_idx] = counts[idx]
            counts = aligned

    null_count = int(series.isna().sum())
    return counts, null_count


def _stable_distribution(counts: np.ndarray, total: int) -> np.ndarray:
    distribution = counts.astype("float64") / float(total)
    return np.clip(distribution, _EPSILON, None)


def compute_feature_psi(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = _DEFAULT_BINS,
) -> tuple[float, pd.DataFrame]:
    """Compute PSI for a single feature.

    Parameters
    ----------
    reference:
        Historical baseline series.
    current:
        Most recent series to compare against the baseline.
    bins:
        Number of quantile-based bins to derive from the reference series.

    Returns
    -------
    tuple[float, pd.DataFrame]
        The scalar PSI value and the per-bin contribution details.
    """

    _validate_inputs(reference, current)
    edges = _resolve_bin_edges(reference, bins)

    ref_counts, ref_nulls = _bin_counts(reference, edges)
    cur_counts, cur_nulls = _bin_counts(current, edges)

    ref_total = int(reference.size)
    cur_total = int(current.size)

    ref_distribution = _stable_distribution(np.append(ref_counts, ref_nulls), ref_total)
    cur_distribution = _stable_distribution(np.append(cur_counts, cur_nulls), cur_total)

    psi_components = (cur_distribution - ref_distribution) * np.log(
        cur_distribution / ref_distribution
    )
    psi_value = float(np.sum(psi_components))

    bin_labels = [
        f"[{edges[idx]:.6g}, {edges[idx + 1]:.6g})" for idx in range(len(edges) - 1)
    ]
    detail = pd.DataFrame(
        {
            "bin": bin_labels + ["<NULL>"],
            "ref_count": np.append(ref_counts, ref_nulls),
            "ref_proportion": ref_distribution,
            "cur_count": np.append(cur_counts, cur_nulls),
            "cur_proportion": cur_distribution,
            "psi_component": psi_components,
        }
    )

    return psi_value, detail


def compute_psi_summary(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    features: Sequence[str],
    *,
    bins: int = _DEFAULT_BINS,
    threshold: float = 0.2,
) -> PSISummary:
    """Compute PSI across a collection of features.

    The returned :class:`PSISummary` exposes a ``feature_psi`` dataframe sorted by
    descending PSI values. A full per-bin breakdown for each feature is attached at
    ``feature_psi.attrs["breakdown"]`` for downstream inspection.
    """

    missing_columns = [
        column
        for column in features
        if column not in reference_frame.columns or column not in current_frame.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns for PSI computation: {missing_columns}")

    rows = []
    breakdowns: list[pd.DataFrame] = []
    for feature in features:
        psi_value, detail = compute_feature_psi(
            reference_frame[feature], current_frame[feature], bins=bins
        )
        rows.append({"feature": feature, "psi": psi_value})
        detail.insert(0, "feature", feature)
        breakdowns.append(detail)

    feature_psi = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    feature_psi["threshold"] = threshold
    feature_psi["breached"] = feature_psi["psi"] >= threshold

    # Attach detailed breakdown for downstream inspection.
    breakdown_frame = pd.concat(breakdowns, ignore_index=True)
    feature_psi.attrs["breakdown"] = breakdown_frame

    return PSISummary(feature_psi=feature_psi, threshold=threshold)

