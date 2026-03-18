"""
Core module for ResXR pipeline.

Provides data structures, constants, configuration, and logging.
"""

from .config import (
    BIDSConfig,
    DeviceConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    ReferenceFrameConfig,
    ReportConfig,
    SessionMapping,
    ValidationConfig,
)
from .constants import (
    BIDS_CHANNEL_PATTERNS,
    BIDS_CHANNEL_TYPE_COUNTS,
    COLUMN_SUFFIXES,
    SYSTEM_COLUMN_PREFIXES,
    TrackingSystem,
)
from .exceptions import (
    BIDSWriteError,
    ColumnMappingError,
    ConfigurationError,
    DataLoadError,
    MissingDataError,
    ResXRError,
    ValidationError,
)
from .logger import get_logger, setup_logging
from .session import (
    QualityFlag,
    Session,
    SessionMetadata,
    TrackingStream,
)

__all__ = [
    # Constants
    "TrackingSystem",
    "SYSTEM_COLUMN_PREFIXES",
    "BIDS_CHANNEL_PATTERNS",
    "COLUMN_SUFFIXES",
    "BIDS_CHANNEL_TYPE_COUNTS",
    # Session/Data
    "Session",
    "SessionMetadata",
    "TrackingStream",
    "QualityFlag",
    # Config
    "PipelineConfig",
    "InputConfig",
    "OutputConfig",
    "SessionMapping",
    "ValidationConfig",
    "PreprocessingConfig",
    "ReportConfig",
    "DeviceConfig",
    "BIDSConfig",
    "ReferenceFrameConfig",
    # Exceptions
    "ResXRError",
    "ConfigurationError",
    "DataLoadError",
    "MissingDataError",
    "ValidationError",
    "BIDSWriteError",
    "ColumnMappingError",
    # Logging
    "setup_logging",
    "get_logger",
]
