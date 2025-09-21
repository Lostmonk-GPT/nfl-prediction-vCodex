"""DuckDB client helper providing a minimal, typed interface."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
from pandas import DataFrame


_TABLE_MODES = {"create", "replace", "append"}


@dataclass(slots=True)
class DuckDBClient(AbstractContextManager["DuckDBClient"]):
    """Context manager wrapper around a DuckDB connection."""

    db_path: str
    read_only: bool = False
    _connection: duckdb.DuckDBPyConnection | None = field(init=False, default=None, repr=False)

    def __enter__(self) -> "DuckDBClient":
        if self._connection is None:
            self._connection = duckdb.connect(database=self.db_path, read_only=self.read_only)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:  # pragma: no cover - exercised indirectly
        self.close()
        return None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            raise RuntimeError("DuckDBClient connection has not been opened.")
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL statement, useful for DDL or imperative commands."""
        return self.connection.execute(sql, params)

    def apply_schema(self, schema_path: str | Path | None = None) -> None:
        """Apply the project schema from disk (defaults to ``schema.sql`` next to this module)."""

        path = Path(schema_path) if schema_path is not None else Path(__file__).with_name("schema.sql")
        schema_sql = path.read_text(encoding="utf-8")
        self.connection.execute(schema_sql)

    def read_sql(self, sql: str, params: Mapping[str, Any] | None = None) -> DataFrame:
        """Run a SQL query and return the results as a pandas ``DataFrame``."""
        result = self.connection.execute(sql, params)
        return result.fetch_df()

    def write_df(self, df: DataFrame, table: str, mode: str = "create") -> None:
        """Persist a ``DataFrame`` into a DuckDB table using the requested mode."""
        if mode not in _TABLE_MODES:
            raise ValueError(f"Unsupported write mode '{mode}'. Expected one of {_TABLE_MODES}.")

        temp_view = f"__df_{uuid4().hex}"
        self.connection.register(temp_view, df)
        identifier = duckdb.escape_identifier(table)

        try:
            if mode == "create":
                self.connection.execute(f"CREATE TABLE {identifier} AS SELECT * FROM {temp_view}")
            elif mode == "replace":
                self.connection.execute(f"CREATE OR REPLACE TABLE {identifier} AS SELECT * FROM {temp_view}")
            else:  # append
                if not self.connection.table_exists(table):
                    raise RuntimeError(f"Table '{table}' does not exist for append mode.")
                self.connection.execute(f"INSERT INTO {identifier} SELECT * FROM {temp_view}")
        finally:
            self.connection.unregister(temp_view)

    def register_parquet(self, path: str, view: str) -> None:
        """Create or replace a view selecting from a Parquet file."""
        identifier = duckdb.escape_identifier(view)
        self.connection.execute(
            f"CREATE OR REPLACE VIEW {identifier} AS SELECT * FROM read_parquet(:path)",
            {"path": path},
        )


__all__ = ["DuckDBClient"]
