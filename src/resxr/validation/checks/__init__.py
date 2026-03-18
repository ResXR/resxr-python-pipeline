"""
Validation checks for ResXR pipeline.

This package contains individual validation check implementations.
"""
# Multi-stream checks: set `required_streams = [TrackingSystem.X, TrackingSystem.Y]`
# to declare dependencies. Access other streams via session.get_stream().

from .eyes_closed import EyesClosedCheck
from .hands_tracking_loss import HandsTrackingLossCheck
from .sampling_rate import SamplingRateCheck
from .stats import StatsSummaryCheck

__all__ = [
    "HandsTrackingLossCheck",
    "SamplingRateCheck",
    "StatsSummaryCheck",
    "EyesClosedCheck",
]
