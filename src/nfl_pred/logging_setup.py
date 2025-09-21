"""Utilities for configuring consistent logging across CLI and pipelines."""

from __future__ import annotations

import logging
import os
from typing import Union


__all__ = ["setup_logging"]


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _coerce_level(value: Union[str, int]) -> int:
    """Translate a string/int log level to the corresponding numeric value."""
    if isinstance(value, int):
        return value

    if not isinstance(value, str):  # pragma: no cover - defensive
        raise TypeError("Log level must be a string or integer")

    candidate = value.strip()
    if not candidate:
        raise ValueError("Log level cannot be empty")

    # Accept integers passed as strings
    try:
        return int(candidate)
    except ValueError:
        pass

    level_name = candidate.upper()
    if level_name in logging._nameToLevel:  # type: ignore[attr-defined]
        return logging._nameToLevel[level_name]  # type: ignore[attr-defined]

    raise ValueError(f"Unknown log level: {value}")


def setup_logging(level: Union[str, int] = "INFO") -> None:
    """Configure application logging.

    Parameters
    ----------
    level:
        Desired log level. Accepts standard logging names (e.g. ``"INFO"``) or
        numeric levels. Overridden by the ``NFLPRED_LOG_LEVEL`` environment
        variable when present.

    Examples
    --------
    >>> from nfl_pred.logging_setup import setup_logging
    >>> import logging
    >>> setup_logging("DEBUG")
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Pipeline ready")
    """

    env_level = os.getenv("NFLPRED_LOG_LEVEL")
    resolved_level = _coerce_level(env_level) if env_level else _coerce_level(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(resolved_level)
            handler.setFormatter(formatter)

    # Ensure library/module loggers use the root configuration
    logging.captureWarnings(True)
