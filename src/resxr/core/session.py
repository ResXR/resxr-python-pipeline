"""
Session and data stream management for the ResXR pipeline.

This module defines the core data structures for holding XR tracking data
throughout the pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..utils import find_first_nonzero_index, find_last_nonzero_index, find_recording_onset
from .constants import TrackingSystem
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityFlag:
    """
    Represents a quality issue detected during validation.

    Quality flags identify time segments with potential data issues that
    may need exclusion or special handling during analysis.
    """

    check_name: str  # Name of the check that generated this flag
    system: TrackingSystem  # Which tracking system is affected
    start_time: float  # Start of flagged segment (seconds)
    end_time: float  # End of flagged segment (seconds)
    severity: str  # "warning" or "error"
    message: str  # Human-readable description
    mask: bool = True  # Whether to mask this segment in preprocessing
    group_name: str | None = None  # Optional human-readable column-group name
    target_columns: list[str] = field(default_factory=list)
    # Empty list means "check applies to all columns in stream" (flag whole row)
    # Non-empty list means "check applies only to these specific columns" (don't flag whole row)
    # Enables column-specific flagging (e.g., only flag left-hand columns when left hand loses tracking)

    def __post_init__(self):
        if self.start_time > self.end_time:
            raise ValueError(
                f"QualityFlag from '{self.check_name}' on {self.system.value} has "
                f"start_time ({self.start_time}) > end_time ({self.end_time})"
            )

    @property
    def duration(self) -> float:
        """Duration of the flagged segment in seconds."""
        return self.end_time - self.start_time

    @classmethod
    def from_mask(
        cls,
        timestamps: np.ndarray,
        boolean_mask: np.ndarray,
        check_name: str,
        system: TrackingSystem,
        severity: str,
        message: str,
        should_mask: bool = True,
        group_name: str | None = None,
        target_columns: list[str] | None = None,
    ) -> list[QualityFlag]:
        """
        Create QualityFlags from a boolean mask by finding contiguous segments.

        Parameters
        ----------
        timestamps : np.ndarray
            Array of timestamps
        boolean_mask : np.ndarray
            Boolean array where True indicates a flagged condition
        check_name : str
            Name of the check that generated this flag
        system : TrackingSystem
            Which tracking system is affected
        severity : str
            "warning" or "error"
        message : str
            Human-readable description
        should_mask : bool
            Whether to mask this segment in preprocessing
        group_name : str | None
            Optional human-readable column-group name for display/reporting
        target_columns : list[str] | None
            Columns this flag targets (empty list = all columns)

        Returns
        -------
        list[QualityFlag]
            List of QualityFlags, one for each contiguous segment
        """
        if target_columns is None:
            target_columns = []

        # Early return if no flagged segments
        if not boolean_mask.any():
            logger.debug(f"No flagged segments found for check '{check_name}' on {system.value}")
            return []

        # Validate inputs
        if len(timestamps) != len(boolean_mask):
            raise ValueError(
                f"timestamps and boolean_mask must have same length: "
                f"{len(timestamps)} != {len(boolean_mask)}"
            )

        # Restrict the mask to the valid recording window (onset → offset)
        # so that leading/trailing zero rows never produce flags.
        onset_idx = find_first_nonzero_index(timestamps)
        offset_idx = find_last_nonzero_index(timestamps)
        if onset_idx is None or offset_idx is None:
            return []
        if onset_idx > 0 or offset_idx < len(boolean_mask) - 1:
            boolean_mask = boolean_mask.copy()
            if onset_idx > 0:
                boolean_mask[:onset_idx] = False
            if offset_idx < len(boolean_mask) - 1:
                boolean_mask[offset_idx + 1 :] = False

        # Find contiguous segments
        segments = cls._find_contiguous_segments(timestamps, boolean_mask)

        # Create flags using list comprehension
        return [
            cls(
                check_name=check_name,
                system=system,
                start_time=start,
                end_time=end,
                severity=severity,
                message=message,
                mask=should_mask,
                group_name=group_name,
                target_columns=target_columns,
            )
            for start, end in segments
        ]

    @staticmethod
    def _find_contiguous_segments(
        timestamps: np.ndarray, mask: np.ndarray
    ) -> list[tuple[float, float]]:
        """
        Find start/end times of contiguous True segments in a boolean mask.

        Uses numpy operations for efficiency.
        """
        if not mask.any():
            return []

        # Find where segments start (transition from False to True)
        # and end (transition from True to False)
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1  # +1: diff[i]==1 means mask[i+1] starts True
        ends = np.where(diff == -1)[0]  # No +1: diff[i]==-1 means mask[i] ends True

        # Handle edge cases
        if mask[0]:  # Segment starts at beginning
            starts = np.concatenate([[0], starts])
        if mask[-1]:  # Segment ends at end
            ends = np.concatenate([ends, [len(mask) - 1]])

        # Ensure we have matching starts and ends
        if len(starts) != len(ends):
            logger.warning(
                "Segment detection produced mismatched starts/ends "
                "(len(starts)=%d, len(ends)=%d); truncating to shorter length. "
                "This may indicate a bug in mask processing.",
                len(starts),
                len(ends),
            )
            min_len = min(len(starts), len(ends))
            starts = starts[:min_len]
            ends = ends[:min_len]

        # Convert to (start_time, end_time) tuples
        return [
            (float(timestamps[start_idx]), float(timestamps[end_idx]))
            for start_idx, end_idx in zip(starts, ends, strict=False)
        ]


@dataclass
class ColumnInfoEntry:
    """One column's BIDS description, parsed from custom_tables.json.

    `description` and `format` are always present. The rest are optional and
    only populated when the source JSON includes them.
    """

    name: str
    description: str
    format: str  # BIDS "Format" field - always present
    units: str | None = None  # numeric columns only
    levels: dict[str, str] | None = None  # categorical columns only
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class CustomTableSchema:
    """Schema for one custom data class, parsed from custom_tables.json."""

    class_name: str
    row_count: int
    columns: list[ColumnInfoEntry]


@dataclass
class SessionMetadata:
    """
    Parsed session metadata from session_metadata.json.

    Contains recording configuration and device information.
    """

    session_id: str
    utc_start: datetime | None = None
    device_utc_offset: str = ""
    unity_version: str = ""
    platform: str = ""
    build_id: str = ""
    ovrplugin_version: str = ""
    sampling_mode: str = ""
    fixed_delta_time: float = 0.02
    schema_rev: str = ""

    # Feature flags indicating which tracking systems were enabled
    face_enabled: bool = False
    body_enabled: bool = False
    hands_enabled: bool = False
    eyes_enabled: bool = False
    controllers_enabled: bool = False

    # Additional metadata
    detected_hand_bones: int = 0
    detected_body_joints: int = 0


@dataclass
class TrackingStream:
    """
    Container for a single tracking system's data.

    Each TrackingStream holds data for one tracking modality (e.g., Head, Hands)
    with its own DataFrame, quality flags, and computed statistics.
    """

    system: TrackingSystem
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    clean_data: pd.DataFrame | None = None
    quality_flags: list[QualityFlag] = field(default_factory=list)

    # Computed metadata
    sampling_frequency: float = 0.0  # Expected frequency
    sampling_frequency_effective: float = 0.0  # Actual measured frequency
    channel_count: int = 0

    # Stats from StatsSummaryCheck (populated during validation)
    stats_summary: pd.DataFrame | None = None
    stats_detailed: pd.DataFrame | None = None

    def __post_init__(self):
        # sampling_frequency must be set explicitly from config
        if self.sampling_frequency == 0.0:
            raise ValueError(
                f"sampling_frequency must be set for {self.system.value} stream (from config)"
            )
        if not self.data.empty:
            time_cols = {"timestamp", "timeSinceStartup"}
            self.channel_count = len([c for c in self.data.columns if c not in time_cols])
            self._compute_effective_rate()

    def _start_timestamp(self) -> float | None:
        """First non-zero timestamp (by row order). Uses ``find_recording_onset``."""
        if self.data.empty or "timestamp" not in self.data.columns:
            return None
        return find_recording_onset(self.data["timestamp"].values)

    def _compute_effective_rate(self) -> None:
        """Effective sampling rate from unique timestamps (start = first non-zero)."""
        if "timestamp" not in self.data.columns:
            raise ValueError(f"Stream {self.system.value}: missing 'timestamp' column")
        if len(self.data) < 2:
            raise ValueError(
                f"Stream {self.system.value}: need at least 2 rows to compute effective rate (got {len(self.data)})"
            )
        start = self._start_timestamp()
        if start is None:
            raise ValueError(f"Stream {self.system.value}: no non-zero timestamp found")
        u = np.unique(self.data["timestamp"].values)
        u = u[u >= start]
        if u.size < 2:
            raise ValueError(
                f"Stream {self.system.value}: fewer than 2 unique timestamps from first non-zero"
            )
        total = u[-1] - u[0]
        if total <= 0:
            raise ValueError(
                f"Stream {self.system.value}: invalid timestamp span (total time <= 0)"
            )
        self.sampling_frequency_effective = (u.size - 1) / total

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds from first non-zero timestamp to max."""
        start = self._start_timestamp()
        if start is None or self.data.empty:
            return 0.0
        return float(self.data["timestamp"].max() - start)

    @property
    def row_count(self) -> int:
        """Number of data samples."""
        return len(self.data)

    @property
    def warning_count(self) -> int:
        """Number of warning-level quality flags."""
        return sum(1 for f in self.quality_flags if f.severity == "warning")

    @property
    def error_count(self) -> int:
        """Number of error-level quality flags."""
        return sum(1 for f in self.quality_flags if f.severity == "error")

    @property
    def interval_ms_expected(self) -> float:
        """Expected inter-sample interval in milliseconds."""
        if self.sampling_frequency > 0:
            return 1000.0 / self.sampling_frequency
        return 0.0

    @property
    def interval_ms_effective(self) -> float:
        """Effective (measured) inter-sample interval in milliseconds."""
        if self.sampling_frequency_effective > 0:
            return 1000.0 / self.sampling_frequency_effective
        return 0.0

    def get_output_data(self) -> pd.DataFrame:
        """Get the data to use for output (clean_data if available, else raw)."""
        return self.clean_data if self.clean_data is not None else self.data


@dataclass
class Session:
    """
    Central state object for the ResXR pipeline.

    A Session represents a single recording session containing multiple
    tracking streams (Head, Hands, Eyes, etc.), metadata, and quality flags.
    """

    session_id: str
    subject_id: str = ""  # BIDS sub-XX identifier
    session_label: str = ""  # BIDS ses-XX identifier

    metadata: SessionMetadata = field(default_factory=lambda: SessionMetadata(session_id=""))
    streams: dict[TrackingSystem, TrackingStream] = field(default_factory=dict)

    # Raw source data (before splitting into streams)
    raw_continuous_data: pd.DataFrame | None = None
    raw_face_data: pd.DataFrame | None = None
    raw_events_data: pd.DataFrame | None = None

    # Custom data classes (parsed from custom_tables.json + their CSVs)
    custom_tables: list[CustomTableSchema] | None = None
    custom_tables_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Filled by merge_events (Step 4/7) just before BIDS events are written
    merged_events_data: pd.DataFrame | None = None
    # Session-level validation flags (populated by run_session_level, Step 8)
    session_flags: list[QualityFlag] = field(default_factory=list)

    # Source paths for reference
    source_dir: str | None = None

    @property
    def all_flags(self) -> list[QualityFlag]:
        """Get all quality flags across all streams, sorted by time."""
        flags = []
        for stream in self.streams.values():
            flags.extend(stream.quality_flags)
        return sorted(flags, key=lambda f: f.start_time)

    @property
    def total_duration_seconds(self) -> float:
        """Maximum duration across all streams."""
        if not self.streams:
            return 0.0
        return max(s.duration_seconds for s in self.streams.values())

    @property
    def total_warning_count(self) -> int:
        """Total warning flags across all streams."""
        return sum(s.warning_count for s in self.streams.values())

    @property
    def total_error_count(self) -> int:
        """Total error flags across all streams."""
        return sum(s.error_count for s in self.streams.values())

    @property
    def masked_time_seconds(self) -> float:
        """Total time marked for masking across all flags.

        Overlapping flag segments are merged before summing, so overlapping
        flags are not double-counted.
        """
        intervals = [(f.start_time, f.end_time) for f in self.all_flags if f.mask]
        if not intervals:
            return 0.0
        intervals.sort(key=lambda x: x[0])
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(end - start for start, end in merged)

    @property
    def masked_percentage(self) -> float:
        """Percentage of total time marked for masking."""
        total = self.total_duration_seconds
        if total == 0:
            return 0.0
        return (self.masked_time_seconds / total) * 100

    def get_stream(self, system: TrackingSystem) -> TrackingStream | None:
        """Get a tracking stream by system type."""
        return self.streams.get(system)

    def has_stream(self, system: TrackingSystem) -> bool:
        """Check if a tracking stream exists."""
        return system in self.streams

    def __repr__(self) -> str:
        streams_info = ", ".join(f"{s.system.value}:{s.row_count}" for s in self.streams.values())
        return (
            f"Session(id='{self.session_id}', "
            f"sub='{self.subject_id}', ses='{self.session_label}', "
            f"streams=[{streams_info}])"
        )
