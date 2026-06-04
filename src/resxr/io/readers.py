"""
Data file readers for the ResXR pipeline.

Handles loading CSV tracking data and JSON metadata files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.config import InputConfig
from ..core.exceptions import DataLoadError, MissingDataError
from ..core.logger import get_logger
from ..core.session import ColumnInfoEntry, CustomTableSchema, Session, SessionMetadata

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
        with open(json_path, encoding="utf-8-sig") as f:
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


@dataclass
class SessionFiles:
    """Resolved paths for one session directory's primary data files."""

    continuous: Path | None
    face: Path | None
    metadata: Path | None
    events: Path | None


def load_custom_tables_json(json_path: Path) -> list[CustomTableSchema] | None:
    """Parse a CustomTables sidecar into a CustomTableSchema list.

    Expects the Unity sourcedata format: a ``"CustomTables"`` object keyed by
    class name; each table carries ``"RowCount"`` and a ``"Columns"`` object
    keyed by column name, whose values hold PascalCase metadata fields
    (``Description``, ``Format`` required; ``Units``, ``Levels``, ``Minimum``,
    ``Maximum`` optional). Column-name keys are taken verbatim (they mirror the
    CSV headers).

    Returns None if the file is missing or unparseable (all-or-nothing: one bad
    entry voids the whole file). Logs an error on any failure so it is findable.
    """
    if not json_path.exists():
        logger.error("CustomTables.json not found at %s", json_path)
        return None
    try:
        with open(json_path, encoding="utf-8-sig") as f:
            data = json.load(f)
        custom_tables = []
        for class_name, table in data["CustomTables"].items():
            cols = [
                ColumnInfoEntry(
                    name=col_name,
                    description=meta["Description"],
                    format=meta["Format"],
                    units=meta.get("Units"),
                    levels=meta.get("Levels"),
                    minimum=meta.get("Minimum"),
                    maximum=meta.get("Maximum"),
                )
                for col_name, meta in table.get("Columns", {}).items()
            ]
            custom_tables.append(
                CustomTableSchema(
                    class_name=class_name,
                    row_count=table.get("RowCount", 0),
                    columns=cols,
                )
            )
        return custom_tables
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        logger.error("Failed to parse CustomTables.json at %s: %s", json_path, e)
        return None


def find_custom_class_csvs(custom_dir: Path, recording_id: str) -> dict[str, Path]:
    """Map each custom data-class CSV in *custom_dir* to its class name.

    The class name is the file stem with a leading ``"{recording_id}_"`` prefix
    removed. Returns ``{}`` when *custom_dir* does not exist.
    """
    if not custom_dir.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(custom_dir.glob("*.csv")):
        class_name = path.stem.removeprefix(f"{recording_id}_")
        result[class_name] = path
    return result


def load_custom_class_csv(path: Path) -> pd.DataFrame:
    """Load one custom-class CSV. Requires numeric lowercase onset & duration."""
    try:
        df = pd.read_csv(
            path,
            na_values=["", "NaN", "null", "None"],
            low_memory=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        raise DataLoadError(f"Failed to load custom class CSV {path}: {e}") from e

    for col in ("onset", "duration"):
        if col not in df.columns:
            raise DataLoadError(
                f"Custom class CSV {path} is missing required column '{col}'. "
                f"Every CustomDataClass must expose float onset and float duration."
            )
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except Exception as e:
            raise DataLoadError(
                f"Column '{col}' in custom class CSV {path} must be numeric: {e}"
            ) from e
    return df


def find_session_files(session_dir: Path, config: InputConfig) -> SessionFiles:
    """Find data files in a session directory using configured patterns."""

    def _most_recent(pattern: str) -> Path | None:
        files = list(session_dir.glob(pattern))
        if len(files) > 1:
            logger.warning(
                "Multiple files matched '%s' in %s; using most recent by mtime.",
                pattern,
                session_dir,
            )
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    return SessionFiles(
        continuous=_most_recent(config.continuous_data_pattern),
        face=_most_recent(config.face_data_pattern),
        metadata=_most_recent(config.metadata_pattern),
        events=_most_recent(config.events_data_pattern),
    )


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
    files = find_session_files(session_dir, config)

    # Metadata is required
    if files.metadata is None:
        raise MissingDataError(
            f"No metadata file matching '{config.metadata_pattern}' in {session_dir}"
        )
    metadata = load_session_metadata(files.metadata)

    # Load continuous data (required)
    if files.continuous is None:
        raise MissingDataError(
            f"No continuous data file matching '{config.continuous_data_pattern}' in {session_dir}"
        )
    raw_continuous = load_continuous_data(files.continuous)

    # Load face data (optional)
    raw_face = None
    if files.face is not None:
        try:
            raw_face = load_face_data(files.face)
        except DataLoadError as e:
            logger.warning(f"Could not load face data: {e}")

    # Load events data (optional but recommended)
    raw_events = None
    if files.events is not None:
        raw_events = load_events_data(files.events)

    # Custom data classes: a dedicated subfolder holding custom_tables.json and CSVs.
    custom_dir = session_dir / config.custom_tables_dir
    custom_csvs = find_custom_class_csvs(custom_dir, metadata.session_id)
    custom_tables_data = {name: load_custom_class_csv(p) for name, p in custom_csvs.items()}

    custom_tables = None
    if custom_tables_data:
        json_matches = sorted(custom_dir.glob("*CustomTables.json"))
        exact = custom_dir / f"{metadata.session_id}_CustomTables.json"
        json_path = exact if exact.exists() else (json_matches[0] if json_matches else None)
        if len(json_matches) > 1:
            logger.warning(
                "Multiple files matching '*CustomTables.json' in %s (%s); using '%s'.",
                custom_dir,
                [p.name for p in json_matches],
                json_path.name,
            )
        custom_tables = load_custom_tables_json(json_path) if json_path else None
        if custom_tables is None:
            logger.error(
                "Custom class CSVs present in %s but the CustomTables sidecar is missing "
                "or unparseable. Events sidecar will not describe custom columns.",
                custom_dir,
            )
        else:
            for schema in custom_tables:
                df = custom_tables_data.get(schema.class_name)
                if df is None:
                    logger.warning(
                        "custom_tables.json declares class '%s' but no matching CSV was "
                        "loaded in %s.",
                        schema.class_name,
                        custom_dir,
                    )
                elif schema.row_count != len(df):
                    logger.warning(
                        "Custom class '%s' row_count %d does not match CSV row count %d in %s.",
                        schema.class_name,
                        schema.row_count,
                        len(df),
                        custom_dir,
                    )

    # Create session
    session = Session(
        session_id=metadata.session_id,
        metadata=metadata,
        raw_continuous_data=raw_continuous,
        raw_face_data=raw_face,
        raw_events_data=raw_events,
        custom_tables=custom_tables,
        custom_tables_data=custom_tables_data,
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
