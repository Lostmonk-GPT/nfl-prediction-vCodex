"""Configuration loading utilities for nfl_pred."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ENV_PREFIX = "NFLPRED__"


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    duckdb_path: str


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str


@dataclass(frozen=True)
class FeatureWindowsConfig:
    short: int
    mid: int


@dataclass(frozen=True)
class FeaturesConfig:
    windows: FeatureWindowsConfig


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    mlflow: MLflowConfig
    features: FeaturesConfig

    def as_dict(self) -> Mapping[str, Any]:
        """Return the configuration as a dictionary for downstream use."""
        return dataclasses.asdict(self)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load configuration from YAML and apply environment overrides."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    try:
        raw_config = _load_yaml(config_path)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Invalid YAML in config file '{config_path}': {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError(
            f"Configuration file '{config_path}' must contain a mapping at the root."
        )

    merged_config = _apply_env_overrides(raw_config)
    return _build_config(merged_config)


def dump_config(config: Config) -> str:
    """Return a YAML string of the effective configuration for debugging."""
    return yaml.safe_dump(config.as_dict(), sort_keys=False)


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Configuration file '{path}' does not exist.")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        raise ConfigError(f"Configuration file '{path}' is empty.")

    return data


def _apply_env_overrides(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(raw_config)

    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue

        path_keys = [segment.lower() for segment in key[len(ENV_PREFIX) :].split("__") if segment]
        if not path_keys:
            continue

        _set_nested_value(config, path_keys, _parse_env_value(value))

    return config


def _set_nested_value(config: dict[str, Any], keys: list[str], value: Any) -> None:
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def _parse_env_value(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return value
    return parsed


def _build_config(data: Mapping[str, Any]) -> Config:
    try:
        paths_cfg = PathsConfig(**_expect_mapping(data, "paths"))
        mlflow_cfg = MLflowConfig(**_expect_mapping(data, "mlflow"))

        features_data = _expect_mapping(data, "features")
        windows_cfg = FeatureWindowsConfig(**_expect_mapping(features_data, "windows"))
        features_cfg = FeaturesConfig(windows=windows_cfg)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Configuration structure invalid: {exc}") from exc

    return Config(paths=paths_cfg, mlflow=mlflow_cfg, features=features_cfg)


def _expect_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    try:
        value = data[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Missing required configuration section '{key}'.") from exc

    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section '{key}' must be a mapping.")

    return value


__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "dump_config",
    "load_config",
]
