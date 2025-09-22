"""Audit trail writer for predictions and feature snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from nfl_pred.storage.duckdb_client import DuckDBClient


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """Representation of a persisted audit trail entry."""

    season: int
    week: int
    snapshot_at: datetime
    model_id: str
    asof_ts: datetime | None
    model_hash: str
    code_version: str
    feature_spec_checksum: str
    upstream_versions: Mapping[str, Any]
    input_row_hashes: tuple[str, ...]
    input_rows_digest: str
    input_row_count: int
    metadata: Mapping[str, Any] | None = None


def gather_audit_record(
    *,
    season: int,
    week: int,
    snapshot_at: datetime,
    model_id: str,
    dataset_versions: Mapping[str, Any],
    code_version: str,
    feature_spec: os.PathLike[str] | str | bytes,
    model_artifact: os.PathLike[str] | str | bytes | None,
    input_rows: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    asof_ts: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AuditRecord:
    """Build an :class:`AuditRecord` from raw runtime artifacts."""

    normalized_snapshot = _coerce_datetime(snapshot_at)
    normalized_asof = _coerce_datetime(asof_ts) if asof_ts is not None else None

    normalized_versions = _normalize_mapping(dataset_versions)
    normalized_metadata = _normalize_mapping(metadata) if metadata is not None else None

    feature_spec_checksum = _resolve_artifact_checksum(feature_spec)
    model_hash = _resolve_artifact_checksum(model_artifact) if model_artifact is not None else "unknown"

    row_hashes, combined_digest = _compute_input_hashes(input_rows)

    return AuditRecord(
        season=int(season),
        week=int(week),
        snapshot_at=normalized_snapshot,
        model_id=str(model_id),
        asof_ts=normalized_asof,
        model_hash=model_hash,
        code_version=str(code_version),
        feature_spec_checksum=feature_spec_checksum,
        upstream_versions=normalized_versions,
        input_row_hashes=tuple(row_hashes),
        input_rows_digest=combined_digest,
        input_row_count=len(row_hashes),
        metadata=normalized_metadata,
    )


def write_audit_record(
    record: AuditRecord,
    *,
    duckdb_path: str | os.PathLike[str],
    ensure_schema: bool = True,
) -> None:
    """Persist ``record`` into the DuckDB ``audit`` table."""

    payload = _serialize_record(record)
    frame = pd.DataFrame([payload])

    with DuckDBClient(str(duckdb_path)) as client:
        if ensure_schema:
            client.apply_schema()

        client.execute(
            """
            DELETE FROM audit
            WHERE season = ?
              AND week = ?
              AND snapshot_at = ?
              AND model_id = ?
            """,
            (
                record.season,
                record.week,
                record.snapshot_at,
                record.model_id,
            ),
        )

        try:
            client.write_df(frame, "audit", mode="append")
        except RuntimeError as error:
            message = str(error).lower()
            if "does not exist" in message or "table" in message and "exist" in message:
                client.write_df(frame, "audit", mode="create")
            else:
                raise RuntimeError(f"Failed to write audit record: {error}") from error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _normalize_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in sorted(data.items(), key=lambda item: str(item[0])):
        normalized[str(key)] = _normalize_value(value)
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_mapping(value)

    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]

    if isinstance(value, datetime):
        if value.tzinfo is None:
            normalized = value.replace(tzinfo=timezone.utc)
        else:
            normalized = value.astimezone(timezone.utc)
        return normalized.isoformat().replace("+00:00", "Z")

    if isinstance(value, pd.Timestamp):
        return _normalize_value(value.to_pydatetime())

    if isinstance(value, np.ndarray):
        return [_normalize_value(item) for item in value.tolist()]

    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        try:
            return _normalize_value(value.item())
        except Exception:  # pragma: no cover - defensive
            pass

    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (str, bool)):
        return value

    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if pd.isna(value):  # type: ignore[arg-type]
        return None

    return str(value)


def _resolve_artifact_checksum(artifact: os.PathLike[str] | str | bytes) -> str:
    if isinstance(artifact, (str, os.PathLike)):
        path = Path(artifact)
        if not path.exists():
            raise FileNotFoundError(f"Artifact '{path}' does not exist for checksum computation.")
        return _hash_file(path)

    if isinstance(artifact, bytes):
        return hashlib.sha256(artifact).hexdigest()

    raise TypeError(f"Unsupported artifact type: {type(artifact)!r}")


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _compute_input_hashes(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
) -> tuple[list[str], str]:
    normalized_rows = _normalize_rows(rows)
    row_hashes = [_hash_json(row) for row in normalized_rows]
    row_hashes.sort()

    combined = _hash_json({"rows": row_hashes, "count": len(row_hashes)})
    return row_hashes, combined


def _normalize_rows(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if rows is None:
        return []

    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)

    normalized: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Input rows must be mappings or DataFrame records.")
        normalized.append(_normalize_mapping(record))
    return normalized


def _hash_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _serialize_record(record: AuditRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "season": record.season,
        "week": record.week,
        "snapshot_at": record.snapshot_at,
        "model_id": record.model_id,
        "asof_ts": record.asof_ts,
        "model_hash": record.model_hash,
        "code_version": record.code_version,
        "feature_spec_checksum": record.feature_spec_checksum,
        "upstream_versions_json": json.dumps(
            record.upstream_versions, sort_keys=True, separators=(",", ":")
        ),
        "input_rows_hash": record.input_rows_digest,
        "input_row_hashes_json": json.dumps(
            list(record.input_row_hashes), sort_keys=True, separators=(",", ":")
        ),
        "input_row_count": record.input_row_count,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }

    if record.metadata is not None:
        payload["metadata_json"] = json.dumps(
            record.metadata, sort_keys=True, separators=(",", ":")
        )
    else:
        payload["metadata_json"] = None

    return payload


__all__ = ["AuditRecord", "gather_audit_record", "write_audit_record"]
