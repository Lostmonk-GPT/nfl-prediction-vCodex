"""Unit tests for SHAP explainability helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from nfl_pred.explain import ShapConfig, generate_shap_artifacts


def _train_simple_tree() -> DecisionTreeClassifier:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "feature_a": rng.normal(size=200),
            "feature_b": rng.uniform(-1, 1, size=200),
        }
    )
    labels = (features["feature_a"] + features["feature_b"] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(features, labels)
    return model


def test_generate_shap_artifacts_creates_outputs(tmp_path: Path) -> None:
    model = _train_simple_tree()
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "feature_a": rng.normal(size=100),
            "feature_b": rng.uniform(-1, 1, size=100),
        }
    )

    config = ShapConfig(sample_fraction=0.2, random_state=1, output_dir=tmp_path)

    artifacts = generate_shap_artifacts(model, features, config=config, prefix="test")

    assert artifacts.values_path.exists()
    assert artifacts.plot_paths
    for path in artifacts.plot_paths.values():
        assert path.exists()

    saved = pd.read_parquet(artifacts.values_path)
    # 20% of 100 rows = 20 samples.
    assert saved.shape[0] == 20
    assert {"feature_a", "feature_b", "shap_value", "shap_base_value"}.issubset(saved.columns)


def test_generate_shap_artifacts_respects_max_samples(tmp_path: Path) -> None:
    model = _train_simple_tree()
    rng = np.random.default_rng(123)
    features = pd.DataFrame(
        {
            "feature_a": rng.normal(size=50),
            "feature_b": rng.normal(size=50),
        }
    )

    config = ShapConfig(sample_fraction=0.5, max_samples=5, random_state=2, output_dir=tmp_path)

    artifacts = generate_shap_artifacts(model, features, config=config, prefix="limited")

    saved = pd.read_parquet(artifacts.values_path)
    assert saved.shape[0] == 5
