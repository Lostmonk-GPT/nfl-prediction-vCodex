"""Utilities for persisting raw weather API payloads with metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def _slugify(component: str) -> str:
    """Return a filesystem-safe representation of ``component``."""

    trimmed = component.strip().strip("/")
    if not trimmed:
        return "segment"

    safe_chars = []
    for char in trimmed:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    slug = "".join(safe_chars).strip("_")
    return slug or "segment"


def _normalize_value(value: Any) -> Any:
    """Normalize ``value`` so that JSON serialization is deterministic."""

    if isinstance(value, Mapping):
        return {str(key): _normalize_value(sub_value) for key, sub_value in sorted(value.items())}

    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]

    if isinstance(value, datetime):
        as_utc = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return as_utc.isoformat().replace("+00:00", "Z")

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[no-any-return]
        except TypeError:
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _normalize_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a deterministic JSON-serializable mapping for ``params``."""

    if not params:
        return {}

    return {str(key): _normalize_value(value) for key, value in sorted(params.items())}


def build_artifact_key(source: str, endpoint: str, params: Mapping[str, Any] | None = None) -> str:
    """Return a stable key derived from ``endpoint`` and ``params``."""

    normalized_params = _normalize_params(params)
    params_blob = json.dumps(normalized_params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(params_blob.encode("utf-8")).hexdigest()[:16]

    source_slug = _slugify(source)
    endpoint_components = [_slugify(part) for part in endpoint.split("/") if part]
    components = [source_slug, *endpoint_components, digest]
    return "/".join(components)


@dataclass(frozen=True)
class WeatherArtifactMetadata:
    """Metadata describing a persisted weather payload."""

    key: str
    source: str
    endpoint: str
    params: Mapping[str, Any]
    called_at: datetime
    version: str
    ttl_expires_at: datetime | None
    payload_path: Path


@dataclass(frozen=True)
class WeatherArtifact:
    """Container bundling a payload with its metadata."""

    payload: Mapping[str, Any]
    metadata: WeatherArtifactMetadata


class WeatherArtifactStore:
    """Filesystem-backed store for raw weather API payloads."""

    def __init__(self, base_dir: str | Path = "data/weather") -> None:
        self._base_dir = Path(base_dir)
        self._manifest_path = self._base_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save(
        self,
        *,
        source: str,
        endpoint: str,
        params: Mapping[str, Any] | None,
        payload: Mapping[str, Any],
        version: str = "unknown",
        called_at: datetime | None = None,
        ttl_seconds: float | None = None,
    ) -> WeatherArtifactMetadata:
        """Persist ``payload`` along with metadata and return the metadata."""

        self._base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = called_at.astimezone(UTC) if called_at else datetime.now(tz=UTC)
        normalized_params = _normalize_params(params)
        key = build_artifact_key(source, endpoint, normalized_params)
        relative_payload_path = Path(*key.split("/"))
        payload_path = (self._base_dir / relative_payload_path).with_suffix(".json")
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        ttl_expires_at = None
        if ttl_seconds is not None:
            ttl_expires_at = timestamp + timedelta(seconds=float(ttl_seconds))

        metadata_record = {
            "key": key,
            "source": source,
            "endpoint": endpoint,
            "params": normalized_params,
            "called_at": timestamp.isoformat().replace("+00:00", "Z"),
            "version": version,
            "ttl_expires_at": (
                ttl_expires_at.isoformat().replace("+00:00", "Z")
                if ttl_expires_at is not None
                else None
            ),
            "payload_path": str(relative_payload_path.with_suffix(".json")),
        }

        _write_json(payload_path, payload)

        manifest = self._read_manifest()
        manifest[key] = metadata_record
        self._write_manifest(manifest)

        metadata = WeatherArtifactMetadata(
            key=key,
            source=source,
            endpoint=endpoint,
            params=normalized_params,
            called_at=timestamp,
            version=version,
            ttl_expires_at=ttl_expires_at,
            payload_path=payload_path,
        )

        return metadata

    def load(
        self,
        source: str,
        endpoint: str,
        params: Mapping[str, Any] | None,
    ) -> WeatherArtifact | None:
        """Load an artifact for ``source``/``endpoint``/``params`` if present."""

        normalized_params = _normalize_params(params)
        key = build_artifact_key(source, endpoint, normalized_params)
        return self.load_by_key(key)

    def load_by_key(self, key: str) -> WeatherArtifact | None:
        """Return the artifact associated with ``key`` if it exists."""

        manifest = self._read_manifest()
        record = manifest.get(key)
        if not record:
            return None

        payload_path = self._base_dir / record["payload_path"]
        if not payload_path.exists():
            return None

        payload = _read_json(payload_path)
        metadata = WeatherArtifactMetadata(
            key=record["key"],
            source=record["source"],
            endpoint=record["endpoint"],
            params=record["params"],
            called_at=_parse_datetime(record["called_at"]),
            version=record.get("version", "unknown"),
            ttl_expires_at=_parse_datetime(record.get("ttl_expires_at")),
            payload_path=payload_path,
        )
        return WeatherArtifact(payload=payload, metadata=metadata)

    def list_keys(self) -> list[str]:
        """Return the stored artifact keys in insertion order."""

        manifest = self._read_manifest()
        return list(manifest.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_manifest(self) -> MutableMapping[str, MutableMapping[str, Any]]:
        if not self._manifest_path.exists():
            return {}

        with self._manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "entries" in data:
            entries = data.get("entries")
            if isinstance(entries, dict):
                return entries

        if isinstance(data, dict):
            return data  # Backward compatibility for plain dict manifests

        raise ValueError("Manifest file is corrupt or malformed")

    def _write_manifest(self, manifest: Mapping[str, Mapping[str, Any]]) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": manifest}
        _write_json(self._manifest_path, data)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(_prepare_payload(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_datetime(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None

    if isinstance(raw, datetime):
        return raw.astimezone(UTC)

    text = str(raw)
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid datetime format: {raw}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _prepare_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _prepare_payload(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_prepare_payload(item) for item in value]

    if isinstance(value, datetime):
        converted = value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return converted.isoformat().replace("+00:00", "Z")

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)

