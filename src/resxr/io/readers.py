"""
Data file readers for the ResXR pipeline.

Handles loading CSV tracking data and JSON metadata files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.config import InputConfig
from ..core.exceptions import DataLoadError, MissingDataError
from ..core.logger import get_logger
from ..core.session import Session, SessionMetadata

logger = get_logger(__name__)


def load_continuous_data(csv_path: Path) -> pd.DataFrame:
    """
    Load ContinuousData CSV with proper dtype handling.

    Parameters
    ----------
    csv_path : Path
        Path to ContinuousData CSV file

    Returns
    -------
    pd.DataFrame
        Loaded data with 'timestamp' column renamed from 'timeSinceStartup'

    Raises
    ------
    DataLoadError
        If file cannot be read or parsed
    """
    logger.info(f"Loading continuous data from: {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            na_values=["", "NaN", "null", "None"],
            low_memory=False,
            encoding="utf-8-sig",  # Handle BOM
        )
    except Exception as e:
        raise DataLoadError(f"Failed to load {csv_path}: {e}") from e

    # Rename timestamp column for consistency
    if "timeSinceStartup" in df.columns:
        df = df.rename(columns={"timeSinceStartup": "timestamp"})
    elif "\ufefftimeSinceStartup" in df.columns:
        # Handle BOM in column name
        df = df.rename(columns={"\ufefftimeSinceStartup": "timestamp"})

    if "timestamp" not in df.columns:
        raise DataLoadError(f"No timestamp column found in {csv_path}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_face_data(csv_path: Path) -> pd.DataFrame:
    """
    Load FaceExpressionData CSV with proper boolean/float handling.

    Parameters
    ----------
    csv_path : Path
        Path to FaceExpressionData CSV file

    Returns
    -------
    pd.DataFrame
        Loaded face expression data

    Raises
    ------
    DataLoadError
        If file cannot be read or parsed
    """
    logger.info(f"Loading face data from: {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            na_values=["", "NaN", "null", "None"],
            low_memory=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        raise DataLoadError(f"Failed to load {csv_path}: {e}") from e

    # Rename timestamp column
    if "timeSinceStartup" in df.columns:
        df = df.rename(columns={"timeSinceStartup": "timestamp"})
    elif "\ufefftimeSinceStartup" in df.columns:
        df = df.rename(columns={"\ufefftimeSinceStartup": "timestamp"})

    if "Face_Status" in df.columns:
        if df["Face_Status"].dtype == object:
            df["Face_Status"] = df["Face_Status"].str.strip()
        was_na = df["Face_Status"].isna()
        df["Face_Status"] = df["Face_Status"].map(
            {
                "true": True,
                "True": True,
                "TRUE": True,
                True: True,
                "1": True,
                "false": False,
                "False": False,
                "FALSE": False,
                False: False,
                "0": False,
            }
        )
        new_na = df["Face_Status"].isna() & ~was_na
        if new_na.any():
            logger.warning(
                "Face_Status contains %d unmapped values that became NaN. "
                "Check the source data for unexpected representations.",
                new_na.sum(),
            )

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_events_data(csv_path: Path) -> pd.DataFrame:
    """
    Load events CSV file with BIDS-compliant column naming.

    Parameters
    ----------
    csv_path : Path
        Path to events CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: name, onset, duration

    Raises
    ------
    DataLoadError
        If file cannot be read, required columns are missing, or dtypes are invalid
    """
    logger.info(f"Loading events data from: {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            na_values=["", "NaN", "null", "None"],
            low_memory=False,
            encoding="utf-8-sig",  # Handle BOM
        )
    except Exception as e:
        raise DataLoadError(f"Failed to load events from {csv_path}: {e}") from e

    required_cols = {"name", "onset", "duration"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise DataLoadError(
            f"Events file {csv_path} is missing required columns: {sorted(missing)}"
        )

    # Ensure numeric onset and duration as float
    for col in ["onset", "duration"]:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except Exception as e:
            raise DataLoadError(
                f"Column '{col}' in events file {csv_path} must be numeric: {e}"
            ) from e

    # Sort by onset time
    df = df.sort_values("onset").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} events from {csv_path}")
    return df


def load_session_metadata(json_path: Path) -> SessionMetadata:
    """
    Parse session_metadata.json into SessionMetadata dataclass.

    Parameters
    ----------
    json_path : Path
        Path to session_metadata.json file

    Returns
    -------
    SessionMetadata
        Parsed metadata

    Raises
    ------
    DataLoadError
        If file cannot be read or parsed
    """
    logger.info(f"Loading metadata from: {json_path}")

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise DataLoadError(f"Failed to load {json_path}: {e}") from e

    # Parse UTC start time
    utc_start = None
    utc_str = data.get("utc_start_iso8601", "")
    if utc_str:
        try:
            # Handle various ISO 8601 formats (fromisoformat supports both with/without fractional seconds)
            utc_str = utc_str.rstrip("Z")
            utc_start = datetime.fromisoformat(utc_str)
        except ValueError:
            logger.warning(f"Could not parse UTC timestamp: {utc_str}")

    return SessionMetadata(
        session_id=data.get("session_id", "unknown"),
        utc_start=utc_start,
        device_utc_offset=data.get("device_utc_offset", ""),
        unity_version=data.get("unity_version", ""),
        platform=data.get("platform", ""),
        build_id=data.get("build_id", ""),
        ovrplugin_version=data.get("ovrplugin_runtime_version", ""),
        sampling_mode=data.get("sampling_mode", ""),
        fixed_delta_time=data.get("fixedDeltaTime", 0.02),
        schema_rev=data.get("schema_rev", ""),
        face_enabled=data.get("face_enabled", False),
        body_enabled=data.get("body_enabled", False),
        hands_enabled=data.get("hands_enabled", False),
        eyes_enabled=data.get("eyes_enabled", False),
        controllers_enabled=data.get("controllers_enabled", False),
        detected_hand_bones=data.get("detected_hand_bones", 0),
        detected_body_joints=data.get("detected_body_joints", 0),
    )


def find_session_files(
    session_dir: Path, config: InputConfig
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """
    Find data files in a session directory using configured patterns.

    Parameters
    ----------
    session_dir : Path
        Directory to search
    config : InputConfig
        Input configuration with file patterns

    Returns
    -------
    tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]
        (metadata_path, continuous_data_path, face_data_path, events_path)
    """
    metadata_files = list(session_dir.glob(config.metadata_pattern))
    continuous_files = list(session_dir.glob(config.continuous_data_pattern))
    face_files = list(session_dir.glob(config.face_data_pattern))
    events_files = list(session_dir.glob(config.events_data_pattern))

    # Take the most recent file if multiple matches
    metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime) if metadata_files else None
    continuous_path = (
        max(continuous_files, key=lambda p: p.stat().st_mtime) if continuous_files else None
    )
    face_path = max(face_files, key=lambda p: p.stat().st_mtime) if face_files else None
    events_path = max(events_files, key=lambda p: p.stat().st_mtime) if events_files else None

    if (
        len(metadata_files) > 1
        or len(continuous_files) > 1
        or len(face_files) > 1
        or len(events_files) > 1
    ):
        logger.warning(
            "Multiple files matched patterns in %s (metadata=%d, continuous=%d, face=%d, events=%d); "
            "using most recent by mtime. Verify the correct file was selected.",
            session_dir,
            len(metadata_files),
            len(continuous_files),
            len(face_files),
            len(events_files),
        )

    return metadata_path, continuous_path, face_path, events_path


def load_session(session_dir: Path, config: InputConfig) -> Session:
    """
    Load all data files from a session directory into a Session object.

    Parameters
    ----------
    session_dir : Path
        Directory containing session data files
    config : InputConfig
        Input configuration specifying file patterns

    Returns
    -------
    Session
        Session object with raw data loaded

    Raises
    ------
    MissingDataError
        If required files are not found
    DataLoadError
        If files cannot be loaded
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise MissingDataError(f"Session directory not found: {session_dir}")

    logger.info(f"Loading session from: {session_dir}")

    # Find files
    metadata_path, continuous_path, face_path, events_path = find_session_files(session_dir, config)

    # Metadata is required
    if metadata_path is None:
        raise MissingDataError(
            f"No metadata file matching '{config.metadata_pattern}' in {session_dir}"
        )
    metadata = load_session_metadata(metadata_path)

    # Load continuous data (required)
    if continuous_path is None:
        raise MissingDataError(
            f"No continuous data file matching '{config.continuous_data_pattern}' in {session_dir}"
        )
    raw_continuous = load_continuous_data(continuous_path)

    # Load face data (optional)
    raw_face = None
    if face_path is not None:
        try:
            raw_face = load_face_data(face_path)
        except DataLoadError as e:
            logger.warning(f"Could not load face data: {e}")

    # Load events data (optional but recommended)
    raw_events = None
    if events_path is not None:
        raw_events = load_events_data(events_path)

    # Create session
    session = Session(
        session_id=metadata.session_id,
        metadata=metadata,
        raw_continuous_data=raw_continuous,
        raw_face_data=raw_face,
        raw_events_data=raw_events,
        source_dir=str(session_dir),
    )

    logger.info(f"Session loaded: {session.session_id}")
    return session


def discover_sessions(config: InputConfig) -> list[Path]:
    """
    Find all session directories in the input data directory.

    Parameters
    ----------
    config : InputConfig
        Input configuration with data_dir path

    Returns
    -------
    List[Path]
        List of directories containing session data
    """
    data_dir = config.data_dir
    if not data_dir.exists():
        raise MissingDataError(f"Data directory not found: {data_dir}")

    sessions = []
    for item in sorted(data_dir.iterdir()):
        if not item.is_dir():
            continue

        # Check if directory contains metadata file
        metadata_files = list(item.glob(config.metadata_pattern))
        if metadata_files:
            sessions.append(item)

    logger.info(f"Discovered {len(sessions)} sessions in {data_dir}")
    return sessions
