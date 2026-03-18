"""
Preprocessing module for ResXR pipeline.

Provides:

- ``preprocess_stream``: optional quality flag masking (NaN replacement)
  for the derivative dataset.
- ``prepare_motion_data``: converts internal time columns (``timestamp``,
  ``timeSinceStartup``) into BIDS LATENCY channels (``latency``,
  ``latency_global``) and strips the originals.  Called at write time
  for both RAW and DERIVATIVE output.
"""

from .stream_preprocessing import (
    prepare_motion_data,
    preprocess_stream,
)

__all__ = [
    "preprocess_stream",
    "prepare_motion_data",
]
