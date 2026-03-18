"""
Custom exceptions for the ResXR pipeline.

All pipeline-specific errors inherit from ResXRError for easy catching.
"""

from __future__ import annotations


class ResXRError(Exception):
    """Base exception for all ResXR pipeline errors."""

    pass


class ConfigurationError(ResXRError):
    """Raised when pipeline configuration is invalid or missing."""

    pass


class DataLoadError(ResXRError):
    """Raised when input data files cannot be loaded or parsed."""

    pass


class MissingDataError(ResXRError):
    """Raised when required data or files are missing."""

    pass


class ValidationError(ResXRError):
    """Raised when data fails critical validation checks."""

    pass


class BIDSWriteError(ResXRError):
    """Raised when BIDS output cannot be written."""

    pass


class ColumnMappingError(ResXRError):
    """Raised when column mapping/splitting fails."""

    pass
