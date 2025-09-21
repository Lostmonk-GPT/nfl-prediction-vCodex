import math

import pandas as pd
import pytest

from nfl_pred.features.travel import compute_travel_features, haversine_miles


def test_haversine_miles_known_distance() -> None:
    # Approximate distance between New York (JFK) and Los Angeles (LAX)
    ny_lat, ny_lon = 40.6413, -73.7781
    la_lat, la_lon = 33.9416, -118.4085

    distance = haversine_miles(ny_lat, ny_lon, la_lat, la_lon)

    assert math.isclose(float(distance), 2475, rel_tol=0.05)


@pytest.fixture
def sample_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "game_id": [
                "2023_01_buf_nyj",
                "2023_02_nyj_dal",
                "2023_03_nyj_mia",
            ],
            "start_time": [
                "2023-09-10T17:00:00Z",
                "2023-09-17T20:25:00Z",
                "2023-09-24T13:30:00Z",
            ],
            "home_team": ["NYJ", "DAL", "NYJ"],
            "away_team": ["BUF", "NYJ", "MIA"],
            "venue_latitude": [40.8135, 32.7473, 51.556],
            "venue_longitude": [-74.0745, -97.0945, -0.2796],
            "neutral_site": [False, False, True],
        }
    )


def test_compute_travel_features_basic(sample_schedule: pd.DataFrame) -> None:
    features = compute_travel_features(sample_schedule)

    assert set(features.columns) == {
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "home_away",
        "start_time",
        "neutral_site",
        "travel_miles",
        "days_since_last",
        "venue_latitude",
        "venue_longitude",
    }

    jets_rows = features[features["team"] == "NYJ"].sort_values("week")

    # Week 1 should have no prior travel or rest context.
    first_game = jets_rows.iloc[0]
    assert math.isnan(first_game["travel_miles"])
    assert math.isnan(first_game["days_since_last"])

    # Week 2: road game from New Jersey to Dallas (~1370 miles).
    second_game = jets_rows.iloc[1]
    expected_distance = haversine_miles(
        jets_rows.iloc[0]["venue_latitude"],
        jets_rows.iloc[0]["venue_longitude"],
        second_game["venue_latitude"],
        second_game["venue_longitude"],
    )
    assert math.isclose(
        float(second_game["travel_miles"]), float(expected_distance), rel_tol=1e-6
    )
    expected_days = (
        second_game["start_time"] - first_game["start_time"]
    ).total_seconds() / 86400.0
    assert math.isclose(float(second_game["days_since_last"]), expected_days, rel_tol=1e-6)

    # Week 3: neutral-site game in London should be flagged for both teams.
    third_game = jets_rows.iloc[2]
    assert bool(third_game["neutral_site"])
    assert third_game["home_away"] == "neutral"
    london_distance = haversine_miles(
        second_game["venue_latitude"],
        second_game["venue_longitude"],
        third_game["venue_latitude"],
        third_game["venue_longitude"],
    )
    assert math.isclose(
        float(third_game["travel_miles"]), float(london_distance), rel_tol=1e-6
    )

    # The opponent should also be flagged neutral for the London game.
    dolphins_row = features[(features["game_id"] == "2023_03_nyj_mia") & (features["team"] == "MIA")]
    assert not dolphins_row.empty
    assert bool(dolphins_row.iloc[0]["neutral_site"])
    assert dolphins_row.iloc[0]["home_away"] == "neutral"


def test_compute_travel_features_missing_coordinates(sample_schedule: pd.DataFrame) -> None:
    schedule = sample_schedule.drop(columns=["venue_latitude", "venue_longitude"])

    team_locations = pd.DataFrame(
        {
            "team": ["NYJ", "DAL", "BUF", "MIA"],
            "latitude": [40.8135, 32.7473, 42.7738, 25.958],
            "longitude": [-74.0745, -97.0945, -78.7868, -80.2389],
        }
    )

    features = compute_travel_features(schedule, team_locations=team_locations)

    jets_week2 = features[(features["team"] == "NYJ") & (features["week"] == 2)]
    assert not jets_week2.empty
    assert math.isclose(
        float(jets_week2.iloc[0]["travel_miles"]),
        float(
            haversine_miles(
                team_locations.loc[team_locations["team"] == "NYJ", "latitude"].iloc[0],
                team_locations.loc[team_locations["team"] == "NYJ", "longitude"].iloc[0],
                team_locations.loc[team_locations["team"] == "DAL", "latitude"].iloc[0],
                team_locations.loc[team_locations["team"] == "DAL", "longitude"].iloc[0],
            )
        ),
        rel_tol=1e-6,
    )

    london_rows = features[features["game_id"] == "2023_03_nyj_mia"]
    assert london_rows["travel_miles"].isna().all()
