import pytest
import typer

from nfl_pred import cli


def test_merge_season_inputs_combines_option_and_args():
    result = cli._merge_season_inputs([2022], [2023, 2024])
    assert result == [2022, 2023, 2024]


def test_merge_season_inputs_requires_value():
    with pytest.raises(typer.BadParameter):
        cli._merge_season_inputs(None, [])
