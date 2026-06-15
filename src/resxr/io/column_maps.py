"""
Column mapping utilities for ResXR pipeline.

Maps input CSV columns to tracking systems and infers BIDS channel
metadata (type, component, units).

LATENCY channel recognition:

    The column names ``latency`` and ``latency_global`` are recognised
    as BIDS ``LATENCY`` type channels with units ``s``.  These columns
    are created by ``prepare_motion_data`` and represent per-sample
    timing in seconds from recording onset.

    ``count_tracked_points`` excludes time-related columns
    (``timestamp``, ``timeSinceStartup``, ``latency``, ``latency_global``)
    because they are not physical tracked points.
"""

from __future__ import annotations

from ..core.constants import (
    BIDS_CHANNEL_PATTERNS,
    BIDS_CHANNEL_TYPE_COUNTS,
    COLUMN_SUFFIXES,
    GLOBAL_CLOCK_COLUMN,
    SYSTEM_COLUMN_PREFIXES,
    TrackingSystem,
)


def get_columns_for_system(
    all_columns: list[str],
    system: TrackingSystem,
    alternate_time_columns: dict[str, str] | None = None,
) -> list[str]:
    """
    Extract column names belonging to a specific tracking system.

    Parameters
    ----------
    all_columns : List[str]
        List of all column names from the DataFrame
    system : TrackingSystem
        Tracking system to filter for
    alternate_time_columns : Dict[str, str] | None
        Optional mapping of system names to exactly one alternate time column name.
        Example: {"Hands": "Node_HandLeft_Time", "Eyes": "Eyes_Time"}

    Returns
    -------
    List[str]
        Column names belonging to this system (always includes 'timestamp' or system-specific time column)
    """
    prefixes = SYSTEM_COLUMN_PREFIXES.get(system, [])

    # Per-system time column as primary, global timestamp kept alongside
    system_name = system.value
    time_col = alternate_time_columns.get(system_name) if alternate_time_columns else None

    cols = [time_col] if time_col and time_col in all_columns else ["timestamp"]

    used_time_cols = {cols[0], "timestamp"}

    for col in all_columns:
        if col in used_time_cols:
            continue

        for prefix in prefixes:
            if col.startswith(prefix) or col == prefix:
                cols.append(col)
                break

    # Always include global timestamp as a data column when using per-system time
    if cols[0] != "timestamp" and "timestamp" in all_columns:
        cols.append("timestamp")

    return cols


def classify_columns_by_system(all_columns: list[str]) -> dict[TrackingSystem, list[str]]:
    """
    Classify all columns by their tracking system.

    Parameters
    ----------
    all_columns : List[str]
        List of all column names

    Returns
    -------
    Dict[TrackingSystem, List[str]]
        Mapping of tracking system to its columns
    """
    result = {}
    for system in TrackingSystem:
        cols = get_columns_for_system(all_columns, system)
        if len(cols) > 1:  # More than just timestamp
            result[system] = cols
    return result


def infer_bids_channel_info(column_name: str) -> tuple[str, str, str]:
    """
    Infer BIDS channel type, component, and units from a column name.

    Parameters
    ----------
    column_name : str
        Column name from the source data

    Returns
    -------
    Tuple[str, str, str]
        (channel_type, component, units)
    """
    # BIDS LATENCY channels (computed latency from recording onset)
    if column_name in ("latency", "latency_global"):
        return "LATENCY", "n/a", "s"

    for suffix, ctype, component, units in BIDS_CHANNEL_PATTERNS:
        if column_name.endswith(suffix):
            return ctype, component, units

    # Default for face expression blend shapes
    if _is_face_blend_shape(column_name):
        return "MISC", "n/a", "normalized"

    return "MISC", "n/a", "n/a"


def _is_face_blend_shape(column_name: str) -> bool:
    """Check if column is a facial blend shape value."""
    # Use Face prefixes from constants (excluding Face_ and FaceRegionConfidence)
    face_prefixes = SYSTEM_COLUMN_PREFIXES[TrackingSystem.FACE]
    return any(column_name.startswith(p) for p in face_prefixes)


def extract_tracked_point(column_name: str) -> str:
    """
    Extract the tracked point name from a column name.

    Removes suffixes like _px, _qx, _Vel_x to get the base point name.

    Parameters
    ----------
    column_name : str
        Full column name

    Returns
    -------
    str
        Base tracked point name
    """
    result = column_name
    for suffix in COLUMN_SUFFIXES:
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break

    return result if result else column_name


def count_channel_types(columns: list[str]) -> dict[str, int]:
    """
    Count channels by BIDS type for a list of columns.

    Parameters
    ----------
    columns : List[str]
        List of column names

    Returns
    -------
    Dict[str, int]
        Count of each channel type (e.g., {"POSChannelCount": 6})
    """
    # Initialize all counts to 0
    type_counts: dict[str, int] = dict.fromkeys(BIDS_CHANNEL_TYPE_COUNTS.values(), 0)

    exclude = {"timestamp", GLOBAL_CLOCK_COLUMN}
    for col in columns:
        if col in exclude:
            continue
        ctype, _, _ = infer_bids_channel_info(col)
        key = BIDS_CHANNEL_TYPE_COUNTS.get(ctype, "MISCChannelCount")
        type_counts[key] += 1

    return type_counts


def count_tracked_points(columns: list[str]) -> int:
    """
    Count unique tracked points in a list of columns.

    Parameters
    ----------
    columns : List[str]
        List of column names

    Returns
    -------
    int
        Number of unique tracked points
    """
    # Exclude time-related columns (not physical tracked points)
    exclude = {"timestamp", GLOBAL_CLOCK_COLUMN, "latency", "latency_global"}
    points = set()
    for col in columns:
        if col in exclude:
            continue
        point = extract_tracked_point(col)
        points.add(point)
    return len(points)
