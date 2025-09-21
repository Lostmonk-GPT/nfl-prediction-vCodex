"""Tests for travel distance and rest-day calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_pred.features.travel import compute_travel_features, haversine_miles


@pytest.mark.parametrize(
    "orig_lat, orig_lon, dest_lat, dest_lon, expected",
    [
        (40.7505, -73.9934, 40.7505, -73.9934, 0.0),  # Same point (MetLife Stadium)
        (40.7128, -74.0060, 34.0522, -118.2437, 2445.0),  # NYC to Los Angeles
    ],
)
def test_haversine_miles_known_pairs(orig_lat, orig_lon, dest_lat, dest_lon, expected):
    distance = haversine_miles(orig_lat, orig_lon, dest_lat, dest_lon)
    # ``haversine_miles`` always returns a numpy array, so coerce to float for assertions.
    distance_value = float(distance)
    assert distance_value == pytest.approx(expected, rel=5e-3, abs=1e-3)


def test_compute_travel_features_short_week_and_neutral_site():
    """Team-level travel metrics handle openers, short weeks, and neutral sites."""

    schedule = pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "week": [1, 2, 3],
            "game_id": [
                "2022_01_TEAMATEAMB",
                "2022_02_TEAMCTEAMA",
                "2022_03_TEAMATEAMD",
            ],
            "start_time": [
                "2022-09-10T17:00:00Z",
                "2022-09-14T17:00:00Z",  # Four-day turnaround (short week)
                "2022-09-25T17:00:00Z",
            ],
            "home_team": ["TEAM_A", "TEAM_C", "TEAM_A"],
            "away_team": ["TEAM_B", "TEAM_A", "TEAM_D"],
            "site_latitude": [40.8135, 34.0130, 41.8781],
            "site_longitude": [-74.0744, -118.2870, -87.6298],
            "location": ["Home", "Home", "Neutral"],
        }
    )

    features = compute_travel_features(schedule)
    team_a = features[features["team"] == "TEAM_A"].reset_index(drop=True)

    # Week 1 opener: no prior game, so travel and rest metrics are missing.
    assert np.isnan(team_a.loc[0, "travel_miles"])
    assert np.isnan(team_a.loc[0, "days_since_last"])
    assert team_a.loc[0, "home_away"] == "home"
    assert bool(team_a.loc[0, "neutral_site"]) is False

    # Week 2 short week: four days rest and cross-country travel.
    expected_leg_one = float(
        haversine_miles(
            schedule.loc[0, "site_latitude"],
            schedule.loc[0, "site_longitude"],
            schedule.loc[1, "site_latitude"],
            schedule.loc[1, "site_longitude"],
        )
    )
    assert team_a.loc[1, "days_since_last"] == pytest.approx(4.0, abs=1e-6)
    assert team_a.loc[1, "travel_miles"] == pytest.approx(expected_leg_one, rel=1e-6)
    assert team_a.loc[1, "home_away"] == "away"
    assert bool(team_a.loc[1, "neutral_site"]) is False

    # Week 3 neutral-site game: flagged correctly with travel from the prior venue.
    expected_leg_two = float(
        haversine_miles(
            schedule.loc[1, "site_latitude"],
            schedule.loc[1, "site_longitude"],
            schedule.loc[2, "site_latitude"],
            schedule.loc[2, "site_longitude"],
        )
    )
    assert team_a.loc[2, "days_since_last"] == pytest.approx(11.0, abs=1e-6)
    assert team_a.loc[2, "travel_miles"] == pytest.approx(expected_leg_two, rel=1e-6)
    assert team_a.loc[2, "home_away"] == "neutral"
    assert bool(team_a.loc[2, "neutral_site"]) is True

