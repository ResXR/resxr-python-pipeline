"""
IO module for ResXR pipeline.

Handles loading source data, splitting into tracking streams,
and writing output files.
"""

from .column_maps import (
    classify_columns_by_system,
    count_channel_types,
    count_tracked_points,
    extract_tracked_point,
    get_columns_for_system,
    infer_bids_channel_info,
)
from .readers import (
    discover_sessions,
    find_session_files,
    load_continuous_data,
    load_face_data,
    load_session,
    load_session_metadata,
)
from .splitter import (
    is_system_enabled,
    split_continuous_data,
)
from .writers import (
    write_bids_events,
    write_bids_tsv,
    write_channels_tsv,
    write_json,
    write_motion_tsv,
    write_participants_tsv,
    write_scans_tsv,
)

__all__ = [
    # Readers
    "load_continuous_data",
    "load_face_data",
    "load_session_metadata",
    "load_session",
    "discover_sessions",
    "find_session_files",
    # Column maps
    "get_columns_for_system",
    "classify_columns_by_system",
    "infer_bids_channel_info",
    "extract_tracked_point",
    "count_channel_types",
    "count_tracked_points",
    # Splitter
    "split_continuous_data",
    "is_system_enabled",
    # Writers
    "write_bids_tsv",
    "write_bids_events",
    "write_json",
    "write_motion_tsv",
    "write_channels_tsv",
    "write_participants_tsv",
    "write_scans_tsv",
]
