from pathlib import Path

import pandas as pd

from nfl_pred.storage.duckdb_client import DuckDBClient


def test_register_parquet_handles_quoted_paths(tmp_path: Path) -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    parquet_path = tmp_path / "sam'ple.parquet"
    df.to_parquet(parquet_path)

    with DuckDBClient(":memory:") as client:
        client.register_parquet(str(parquet_path), "sample_view")
        result = client.read_sql("SELECT COUNT(*) AS cnt FROM sample_view")

    assert result.loc[0, "cnt"] == len(df)
