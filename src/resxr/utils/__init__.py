"""
Utility functions for ResXR pipeline.

Provides helper functions for time, quaternion, and file operations.

Key utilities:

- ``find_recording_onset``: finds the first non-zero timestamp in an
  array, used as the single source of truth for recording onset across
  the pipeline (``TrackingStream._start_timestamp``,
  ``prepare_motion_data``, and the HTML report).
- ``find_first_nonzero_index``: returns the positional index of the first
  finite non-zero timestamp (the lower bound of the recording window).
- ``find_last_nonzero_index``: returns the positional index of the last
  finite non-zero timestamp (the upper bound of the recording window).
- ``find_internal_zero_blocks``: detects contiguous zero/NaN blocks
  strictly inside the recording window (between onset and offset).
  Leading/trailing edge zeros are excluded — they are device spin-up/down
  padding, not mid-recording dropouts.
"""

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds

    Returns
    -------
    str
        Formatted string (e.g., "1h 23m 45s" or "45.3s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def find_first_nonzero_index(timestamps: np.ndarray) -> int | None:
    """
    Return the positional index of the first finite non-zero timestamp.

    Skips leading zeros *and* NaN values so that a NaN-prefixed stream
    does not produce a NaN onset.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamp values

    Returns
    -------
    int | None
        Index of the first valid (finite, non-zero) timestamp,
        or ``None`` if no such value exists.
    """
    if timestamps.size == 0:
        return None
    valid = np.isfinite(timestamps) & (timestamps != 0)
    idx = np.flatnonzero(valid)
    return int(idx[0]) if idx.size else None


def find_last_nonzero_index(timestamps: np.ndarray) -> int | None:
    """
    Return the positional index of the last finite non-zero timestamp.

    Symmetric counterpart of ``find_first_nonzero_index``.  Detects
    trailing zeros emitted after the device stops producing real timing
    data.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamp values

    Returns
    -------
    int | None
        Index of the last valid (finite, non-zero) timestamp,
        or ``None`` if no such value exists.
    """
    if timestamps.size == 0:
        return None
    valid = np.isfinite(timestamps) & (timestamps != 0)
    idx = np.flatnonzero(valid)
    return int(idx[-1]) if idx.size else None


def find_recording_onset(timestamps: np.ndarray) -> float | None:
    """
    Find the recording onset: first finite non-zero timestamp value.

    Some systems emit zero-valued timestamps before they start producing
    real timing data. This function skips those initial zeros and NaN
    values.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamp values

    Returns
    -------
    float | None
        First valid timestamp, or None if all zeros / NaN / empty
    """
    idx = find_first_nonzero_index(timestamps)
    if idx is None:
        return None
    onset = float(timestamps[idx])
    if onset < 0:
        _logger.warning(
            "Recording onset is negative (%.6f). This may indicate a clock "
            "error or reset. Latency values will be computed from this value.",
            onset,
        )
    return onset


def find_internal_zero_blocks(timestamps: np.ndarray) -> list[tuple[int, int]]:
    """
    Return positional index ranges for contiguous zero (or NaN) blocks
    that fall strictly inside the recording window (onset–offset).

    Leading and trailing zeros are excluded; only mid-recording dropouts
    are returned.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamp values

    Returns
    -------
    list[tuple[int, int]]
        List of (start_idx, end_idx) pairs (inclusive, in original array
        index space) for each internal zero block. Empty list if none.
    """
    onset_idx = find_first_nonzero_index(timestamps)
    offset_idx = find_last_nonzero_index(timestamps)

    if onset_idx is None or offset_idx is None or onset_idx >= offset_idx:
        return []

    # Strictly between onset and offset (neither endpoint can be zero by definition)
    window_start = onset_idx + 1
    window_end = offset_idx  # offset sample is valid; exclude it (slice is exclusive)

    if window_start >= window_end:
        return []

    w = timestamps[window_start:window_end]
    is_bad = ~np.isfinite(w) | (w == 0)

    d = np.diff(is_bad.astype(np.int8), prepend=0, append=0)
    starts = window_start + np.flatnonzero(d == 1)
    ends = window_start + np.flatnonzero(d == -1) - 1

    return list(zip(starts.tolist(), ends.tolist(), strict=True))


__all__ = [
    "format_duration",
    "find_recording_onset",
    "find_first_nonzero_index",
    "find_last_nonzero_index",
    "find_internal_zero_blocks",
]
