"""Targeted tests for stacking utilities and SHAP sampling (AI-306)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from nfl_pred.explain import ShapConfig, compute_shap_values
from nfl_pred.model.splits import time_series_splits
from nfl_pred.model.stacking import StackingEnsemble, generate_out_of_fold_predictions


def _make_stacking_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(123)
    weeks = np.repeat(np.arange(1, 7), 24)
    feature_1 = rng.normal(size=weeks.size)
    feature_2 = rng.normal(loc=0.15 * weeks, scale=0.5, size=weeks.size)
    logits = 0.9 * feature_1 + 0.8 * feature_2
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    outcomes = rng.binomial(1, probabilities)

    features = pd.DataFrame(
        {
            "week": weeks,
            "feature_1": feature_1,
            "feature_2": feature_2,
        }
    )
    return features, outcomes


def test_oof_predictions_align_with_meta_predictions() -> None:
    features, outcomes = _make_stacking_dataset()
    factories = {
        "logit_f1": lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
        "logit_f2": lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
    }

    splits = list(time_series_splits(features, group_col="week", min_train_weeks=2))
    feature_frame = features[["feature_1", "feature_2"]]

    oof = generate_out_of_fold_predictions(factories, feature_frame, outcomes, splits)

    assert oof.shape == (feature_frame.shape[0], len(factories))
    assert list(oof.columns) == list(factories.keys())

    ensemble = StackingEnsemble(factories, meta_max_iter=400, random_state=7)
    fitted = ensemble.fit(feature_frame, outcomes, splits)
    assert fitted is ensemble

    head = feature_frame.head(10)
    meta_features = ensemble.predict_meta_features(head)
    assert meta_features.shape == (len(head), len(factories))
    # Meta learner should emit well-formed probabilities and binary predictions.
    proba = ensemble.predict_proba(head)
    assert proba.shape == (len(head), 2)
    assert np.all((0.0 <= proba) & (proba <= 1.0))
    preds = ensemble.predict(head)
    assert set(np.unique(preds)).issubset({0, 1})


def test_shap_sampling_fraction_and_reproducibility() -> None:
    rng = np.random.default_rng(99)
    features = pd.DataFrame(
        {
            "feature_a": rng.normal(size=80),
            "feature_b": rng.uniform(-1, 1, size=80),
        }
    )
    labels = (features["feature_a"] + 0.6 * features["feature_b"] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=5)
    model.fit(features, labels)

    config = ShapConfig(sample_fraction=0.15, random_state=17)

    result_first = compute_shap_values(model, features, config=config)
    result_second = compute_shap_values(model, features, config=config)

    expected_samples = int(round(len(features) * config.sample_fraction))
    expected_samples = max(1, min(len(features), expected_samples))
    assert result_first.features.shape[0] == expected_samples
    pdt.assert_frame_equal(result_first.features, result_second.features)
