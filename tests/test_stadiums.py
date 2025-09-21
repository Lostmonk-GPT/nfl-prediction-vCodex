import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nfl_pred.ref.stadiums import StadiumsValidationError, load_stadiums


def test_load_stadiums_returns_validated_dataframe():
    df = load_stadiums()

    expected_columns = {
        "venue",
        "teams",
        "lat",
        "lon",
        "tz",
        "altitude_ft",
        "surface",
        "roof",
        "neutral_site",
    }
    assert expected_columns.issubset(set(df.columns))

    assert df["teams"].apply(lambda value: isinstance(value, tuple)).all()
    shared_row = df.loc[df["venue"] == "SoFi Stadium"].iloc[0]
    assert shared_row["teams"] == ("LAC", "LAR")

    assert df["neutral_site"].dtype == bool
    assert not df["neutral_site"].any()

    assert (df["lat"].between(-90, 90)).all()
    assert (df["lon"].between(-180, 180)).all()


def test_load_stadiums_rejects_invalid_time_zone(tmp_path):
    source = pd.read_csv("data/ref/stadiums.csv")
    source.loc[0, "tz"] = "Mars/OlympusMons"

    custom_data_dir = tmp_path / "data"
    custom_data_dir.mkdir()
    ref_dir = custom_data_dir / "ref"
    ref_dir.mkdir()
    source.to_csv(ref_dir / "stadiums.csv", index=False)

    with pytest.raises(StadiumsValidationError):
        load_stadiums(data_dir=custom_data_dir)
