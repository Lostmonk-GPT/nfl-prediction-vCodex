"""Reference data utilities for stadiums and weather context."""

from .stadiums import load_stadiums, StadiumsValidationError

__all__ = [
    "load_stadiums",
    "StadiumsValidationError",
]
