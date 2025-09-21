"""SHAP explainability helpers with MLflow integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mlflow
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ShapConfig:
    """Configuration controlling SHAP computation and artifact persistence."""

    sample_fraction: float = 0.1
    max_samples: int | None = None
    random_state: int = 0
    approximate: bool = False
    check_additivity: bool = False
    output_dir: Path = Path("data/artifacts/shap")
    summary_plot_types: Sequence[str] = ("bar", "beeswarm")
    mlflow_artifact_subdir: str = "shap"
    plot_dpi: int = 150

    def __post_init__(self) -> None:
        if not 0 < self.sample_fraction <= 1:
            msg = "sample_fraction must be within (0, 1]."
            raise ValueError(msg)
        if self.max_samples is not None and self.max_samples <= 0:
            msg = "max_samples must be a positive integer when provided."
            raise ValueError(msg)
        if self.plot_dpi <= 0:
            msg = "plot_dpi must be positive."
            raise ValueError(msg)


@dataclass(slots=True)
class ShapResult:
    """Container for computed SHAP values."""

    features: pd.DataFrame
    shap_values: np.ndarray
    base_value: float


@dataclass(slots=True)
class ShapArtifacts:
    """Paths to persisted SHAP artifacts."""

    values_path: Path
    plot_paths: dict[str, Path]


def compute_shap_values(
    model: object,
    features: pd.DataFrame | np.ndarray,
    *,
    config: ShapConfig | None = None,
) -> ShapResult:
    """Compute SHAP values for a tree-based model using a sampled subset."""

    shap_config = config or ShapConfig()
    feature_frame = _ensure_dataframe(features)
    sampled = _sample_frame(feature_frame, shap_config)

    explainer = _build_tree_explainer(model, shap_config, background=sampled)

    shap_values = _compute_with_fallback(explainer, sampled, shap_config)
    shap_array = _select_positive_class_shap(shap_values, n_features=sampled.shape[1])

    if shap_array.shape != sampled.shape:
        msg = (
            "Computed SHAP values have unexpected shape %s, expected %s."
            % (shap_array.shape, sampled.shape)
        )
        raise ValueError(msg)

    base_value = _extract_base_value(explainer)
    return ShapResult(features=sampled, shap_values=shap_array, base_value=base_value)


def generate_shap_artifacts(
    model: object,
    features: pd.DataFrame | np.ndarray,
    *,
    config: ShapConfig | None = None,
    prefix: str = "model",
) -> ShapArtifacts:
    """Compute SHAP values, persist them, and optionally log to MLflow."""

    shap_result = compute_shap_values(model, features, config=config)
    shap_config = config or ShapConfig()

    output_dir = Path(shap_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values_path = _save_shap_table(shap_result, output_dir=output_dir, prefix=prefix)
    plot_paths = _save_summary_plots(
        shap_result,
        output_dir=output_dir,
        prefix=prefix,
        plot_types=shap_config.summary_plot_types,
        dpi=shap_config.plot_dpi,
    )

    _log_artifacts_to_mlflow(
        [values_path, *plot_paths.values()],
        artifact_subdir=shap_config.mlflow_artifact_subdir,
    )

    return ShapArtifacts(values_path=values_path, plot_paths=plot_paths)


def _ensure_dataframe(features: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()

    array = np.asarray(features)
    column_names = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=column_names)


def _sample_frame(frame: pd.DataFrame, config: ShapConfig) -> pd.DataFrame:
    n_rows = len(frame)
    sample_size = int(round(n_rows * config.sample_fraction))
    sample_size = max(1, sample_size)
    if config.max_samples is not None:
        sample_size = min(sample_size, config.max_samples)
    sample_size = min(sample_size, n_rows)

    LOGGER.info(
        "Sampling %s of %s rows for SHAP computation (fraction=%.3f).",
        sample_size,
        n_rows,
        config.sample_fraction,
    )

    return frame.sample(n=sample_size, random_state=config.random_state)


def _build_tree_explainer(
    model: object,
    config: ShapConfig,
    *,
    background: pd.DataFrame | None,
) -> shap.TreeExplainer:
    kwargs: dict[str, object] = {"model_output": "probability"}
    if background is not None and not background.empty:
        kwargs["data"] = background
        kwargs["feature_perturbation"] = "interventional"
    else:
        kwargs["feature_perturbation"] = "tree_path_dependent"

    if config.approximate:
        kwargs["algorithm"] = "auto"

    return shap.TreeExplainer(model, **kwargs)


def _compute_with_fallback(
    explainer: shap.TreeExplainer,
    frame: pd.DataFrame,
    config: ShapConfig,
) -> object:
    try:
        return explainer.shap_values(frame, check_additivity=config.check_additivity)
    except (RuntimeError, ValueError) as error:
        LOGGER.warning(
            "Primary SHAP computation failed (%s). Falling back to approximate mode.",
            error,
        )
        fallback_explainer = shap.TreeExplainer(
            explainer.model,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
            algorithm="auto",
        )
        return fallback_explainer.shap_values(frame, check_additivity=False)


def _select_positive_class_shap(values: object, *, n_features: int) -> np.ndarray:
    if isinstance(values, shap.Explanation):
        array = np.asarray(values.values, dtype=float)
    elif isinstance(values, list):
        if not values:
            raise ValueError("Received empty SHAP values list from explainer.")
        array = np.asarray(values[-1], dtype=float)
    else:
        array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        # TreeExplainer may return (n_samples, n_features, n_outputs).
        if array.shape[1] != n_features:
            msg = (
                "Unable to align SHAP values with feature columns: expected %s features but "
                "received axis with size %s."
            )
            raise ValueError(msg % (n_features, array.shape[1]))
        return array[:, :, -1]

    raise ValueError("Unsupported SHAP values dimensionality.")


def _extract_base_value(explainer: shap.TreeExplainer) -> float:
    base_value = np.asarray(explainer.expected_value)
    if base_value.ndim == 0:
        return float(base_value)
    if base_value.ndim == 1:
        return float(base_value[-1])
    return float(base_value.reshape(-1)[-1])


def _save_shap_table(
    result: ShapResult,
    *,
    output_dir: Path,
    prefix: str,
) -> Path:
    table = result.features.copy()
    table["shap_value"] = result.shap_values.sum(axis=1)
    table["shap_base_value"] = result.base_value
    path = output_dir / f"{prefix}_shap_values.parquet"
    table.to_parquet(path, index=False)
    return path


def _save_summary_plots(
    result: ShapResult,
    *,
    output_dir: Path,
    prefix: str,
    plot_types: Sequence[str],
    dpi: int,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for plot_type in plot_types:
        shap.summary_plot(
            result.shap_values,
            result.features,
            plot_type=plot_type,
            show=False,
        )
        fig = plt.gcf()
        path = output_dir / f"{prefix}_summary_{plot_type}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[plot_type] = path
    return paths


def _log_artifacts_to_mlflow(paths: Iterable[Path], *, artifact_subdir: str) -> None:
    active_run = mlflow.active_run()
    if active_run is None:
        LOGGER.debug("No active MLflow run found; skipping SHAP artifact logging.")
        return

    for artifact_path in paths:
        mlflow.log_artifact(str(artifact_path), artifact_path=artifact_subdir)
