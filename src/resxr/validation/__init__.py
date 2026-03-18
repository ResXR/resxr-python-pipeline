"""
Validation module for ResXR pipeline.

Provides quality validation checks for XR tracking data.
"""

# Import checks to register them
from .checks import (
    EyesClosedCheck,
    HandsTrackingLossCheck,
    SamplingRateCheck,
    StatsSummaryCheck,
)
from .registry import CheckRegistry, check_registry, register_check

__all__ = [
    "CheckRegistry",
    "check_registry",
    "register_check",
    "HandsTrackingLossCheck",
    "SamplingRateCheck",
    "StatsSummaryCheck",
    "EyesClosedCheck",
]
