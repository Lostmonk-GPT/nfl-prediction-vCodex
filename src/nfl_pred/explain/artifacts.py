"""Deterministic explainability artifact persistence and discovery utilities."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

import mlflow

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ArtifactMetadata:
    """Metadata describing a persisted explainability artifact bundle."""

    season: int
    week: int
    model_id: str
    artifact_type: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str | None = None
    extra: MutableMapping[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        """Serialize metadata to a JSON-serializable dictionary."""

        payload: dict[str, object] = {
            "season": self.season,
            "week": self.week,
            "model_id": self.model_id,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "extra": dict(self.extra),
        }
        return payload

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ArtifactMetadata":
        """Deserialize metadata from a JSON dictionary."""

        created_raw = data.get("created_at")
        if isinstance(created_raw, str):
            created_at = datetime.fromisoformat(created_raw)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            description = str(description)

        extra_field = data.get("extra")
        if isinstance(extra_field, MutableMapping):
            extra = dict(extra_field)
        elif isinstance(extra_field, Mapping):
            extra = dict(extra_field)
        else:
            extra = {}

        return cls(
            season=int(data["season"]),
            week=int(data["week"]),
            model_id=str(data["model_id"]),
            artifact_type=str(data["artifact_type"]),
            created_at=created_at,
            description=description,
            extra=extra,
        )


@dataclass(slots=True)
class ArtifactRecord:
    """Container bundling metadata with the persisted file paths."""

    metadata: ArtifactMetadata
    files: Mapping[str, Path]
    metadata_path: Path


class ExplainabilityArtifactManager:
    """Persist explainability artifacts with deterministic layout and metadata."""

    def __init__(
        self,
        *,
        base_dir: Path | str = Path("data/artifacts/explain"),
        mlflow_artifact_subdir: str = "explain",
    ) -> None:
        self._base_dir = Path(base_dir)
        self._mlflow_subdir = mlflow_artifact_subdir

    @property
    def base_dir(self) -> Path:
        """Root directory where explainability artifacts are stored."""

        return self._base_dir

    def persist(
        self,
        artifact_type: str,
        artifacts: Mapping[str, Path],
        *,
        season: int,
        week: int,
        model_id: str,
        description: str | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> ArtifactRecord:
        """Persist files and register metadata for an explainability artifact bundle."""

        metadata = ArtifactMetadata(
            season=season,
            week=week,
            model_id=model_id,
            artifact_type=artifact_type,
            description=description,
            extra=dict(extra or {}),
        )

        destination = self._bundle_dir(metadata)
        destination.mkdir(parents=True, exist_ok=True)

        copied_paths: dict[str, Path] = {}
        for name, path in artifacts.items():
            source_path = Path(path)
            if not source_path.exists():
                msg = f"Artifact source path does not exist: {source_path}"
                raise FileNotFoundError(msg)
            dest_path = destination / f"{name}{source_path.suffix}"
            shutil.copy2(source_path, dest_path)
            copied_paths[name] = dest_path

        metadata_path = destination / "metadata.json"
        metadata_path.write_text(json.dumps(metadata.to_json(), indent=2, sort_keys=True))

        self._log_to_mlflow(metadata, copied_paths, metadata_path)

        return ArtifactRecord(metadata=metadata, files=copied_paths, metadata_path=metadata_path)

    def discover(
        self,
        *,
        season: int | None = None,
        week: int | None = None,
        model_id: str | None = None,
        artifact_type: str | None = None,
    ) -> list[ArtifactRecord]:
        """Discover persisted artifacts optionally filtered by metadata fields."""

        records: list[ArtifactRecord] = []
        if not self.base_dir.exists():
            return records

        for metadata_path in self.base_dir.rglob("metadata.json"):
            try:
                metadata = self._load_metadata(metadata_path)
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                LOGGER.warning("Failed to parse metadata at %s: %s", metadata_path, error)
                continue

            if season is not None and metadata.season != season:
                continue
            if week is not None and metadata.week != week:
                continue
            if model_id is not None and metadata.model_id != model_id:
                continue
            if artifact_type is not None and metadata.artifact_type != artifact_type:
                continue

            files = self._load_files(metadata_path.parent)
            records.append(ArtifactRecord(metadata=metadata, files=files, metadata_path=metadata_path))

        records.sort(key=lambda record: (record.metadata.created_at, record.metadata.week))
        return records

    def cleanup(
        self,
        *,
        before: datetime | None = None,
        max_per_model: int | None = None,
        predicate: Callable[[ArtifactMetadata], bool] | None = None,
    ) -> list[Path]:
        """Remove persisted artifacts based on retention policies."""

        records = self.discover()
        if not records:
            return []

        to_remove: set[Path] = set()

        if before is not None:
            for record in records:
                if record.metadata.created_at < before:
                    to_remove.add(record.metadata_path.parent)

        if max_per_model is not None and max_per_model > 0:
            grouped: dict[str, list[ArtifactRecord]] = defaultdict(list)
            for record in records:
                grouped[record.metadata.model_id].append(record)
            for group_records in grouped.values():
                sorted_records = sorted(
                    group_records,
                    key=lambda rec: (
                        rec.metadata.season,
                        rec.metadata.week,
                        rec.metadata.created_at,
                    ),
                    reverse=True,
                )
                for record in sorted_records[max_per_model:]:
                    to_remove.add(record.metadata_path.parent)

        if predicate is not None:
            for record in records:
                if predicate(record.metadata):
                    to_remove.add(record.metadata_path.parent)

        removed_paths: list[Path] = []
        for bundle_dir in sorted(to_remove):
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
                removed_paths.append(bundle_dir)

        return removed_paths

    def _log_to_mlflow(
        self,
        metadata: ArtifactMetadata,
        files: Mapping[str, Path],
        metadata_path: Path,
    ) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            LOGGER.debug("No active MLflow run; skipping explainability artifact logging.")
            return

        artifact_dir = self._mlflow_path(metadata)
        mlflow.log_artifact(str(metadata_path), artifact_path=artifact_dir)
        for path in files.values():
            mlflow.log_artifact(str(path), artifact_path=artifact_dir)

    def _bundle_dir(self, metadata: ArtifactMetadata) -> Path:
        return (
            self.base_dir
            / f"season={metadata.season}"
            / f"week={metadata.week:02d}"
            / f"model={metadata.model_id}"
            / metadata.artifact_type
        )

    def _mlflow_path(self, metadata: ArtifactMetadata) -> str:
        return \
            f"{self._mlflow_subdir}/season={metadata.season}/week={metadata.week:02d}/model={metadata.model_id}/{metadata.artifact_type}"

    def _load_metadata(self, metadata_path: Path) -> ArtifactMetadata:
        raw = json.loads(metadata_path.read_text())
        if not isinstance(raw, Mapping):
            msg = f"Metadata at {metadata_path} is not a JSON object."
            raise ValueError(msg)
        return ArtifactMetadata.from_json(raw)

    def _load_files(self, directory: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.name != "metadata.json":
                name = path.stem
                files[name] = path
        return files


__all__ = [
    "ArtifactMetadata",
    "ArtifactRecord",
    "ExplainabilityArtifactManager",
]

