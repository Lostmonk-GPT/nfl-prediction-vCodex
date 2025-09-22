from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nfl_pred.monitoring.psi import PSISummary, compute_feature_psi, compute_psi_summary


def test_compute_feature_psi_detects_shift():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(loc=0.0, scale=1.0, size=10_000))
    current = pd.Series(rng.normal(loc=0.6, scale=1.0, size=10_000))

    psi_value, detail = compute_feature_psi(reference, current, bins=10)

    assert detail["psi_component"].sum() == pytest.approx(psi_value)
    assert psi_value == pytest.approx(0.376, rel=0.05)
    assert detail.loc[detail["bin"] == "<NULL>", "ref_count"].item() == 0
    assert detail.loc[detail["bin"] == "<NULL>", "cur_count"].item() == 0


def test_compute_feature_psi_handles_nulls():
    reference = pd.Series([1, 2, np.nan, 4, 5, np.nan])
    current = pd.Series([1, 2, 3, np.nan, np.nan, 6])

    psi_value, detail = compute_feature_psi(reference, current, bins=4)

    null_row = detail.loc[detail["bin"] == "<NULL>"]
    assert null_row["ref_count"].item() == 2
    assert null_row["cur_count"].item() == 2
    assert psi_value >= 0.0


def test_compute_psi_summary_flags_breaches():
    reference = pd.DataFrame(
        {
            "feature_a": np.linspace(0, 1, 1000),
            "feature_b": np.linspace(0, 1, 1000),
        }
    )
    current = pd.DataFrame(
        {
            "feature_a": np.linspace(0.2, 1.2, 1000),
            "feature_b": np.linspace(0.01, 1.01, 1000),
        }
    )

    summary = compute_psi_summary(reference, current, ["feature_a", "feature_b"], threshold=0.1)

    assert isinstance(summary, PSISummary)
    assert summary.feature_psi.shape[0] == 2
    assert summary.feature_psi["breached"].tolist() == [True, False]
    assert summary.breached_features == ["feature_a"]
    breakdown = summary.feature_psi.attrs["breakdown"]
    assert set(breakdown["feature"].unique()) == {"feature_a", "feature_b"}


