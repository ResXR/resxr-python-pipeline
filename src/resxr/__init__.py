"""
ResXR - VR Experiment Data Processing Pipeline

An open-source toolkit for standardized XR behavioral research.
Converts Unity/Meta Quest tracking data to BIDS-compliant format.
"""

__version__ = "0.1.0"

from .core import (
    PipelineConfig,
    QualityFlag,
    Session,
    SessionMetadata,
    TrackingStream,
    TrackingSystem,
)
from .pipeline import run

__all__ = [
    "__version__",
    "run",
    "TrackingSystem",
    "Session",
    "SessionMetadata",
    "TrackingStream",
    "QualityFlag",
    "PipelineConfig",
]
