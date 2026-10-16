"""
Data splitting for ResXR pipeline.

Splits the monolithic ContinuousData CSV into per-tracking-system
DataFrames.  This module is **split-only** — it performs no data
transformations beyond column selection and time-column renaming.

Time column handling:

- If an alternate time column is configured for a system (e.g.,
  ``Node_HandLeft_Time`` for Hands), it is renamed to ``timestamp``
  for internal pipeline consistency, and the original global
  ``timestamp`` column is preserved as ``timeSinceStartup``.
- If no alternate is configured, ``timestamp`` remains as-is and
  ``timeSinceStartup`` is absent from the stream.
- A configured clock column that is used is checked for milliseconds and
  backward steps (``readers.check_stream_clock``), with a warning naming
  the column; its values are not changed.

BIDS LATENCY channels are **not** computed here.  That is handled
downstream by ``prepare_motion_data`` in the preprocessing module,
which runs at write time.
"""

from __future__ import annotations

from ..core.constants import GLOBAL_CLOCK_COLUMN, SYSTEM_COLUMN_PREFIXES, TrackingSystem
from ..core.exceptions import ConfigurationError
from ..core.logger import get_logger
from ..core.session import Session, SessionMetadata, TrackingStream
from ..utils import preview
from .column_maps import get_columns_for_system
from .readers import check_stream_clock

logger = get_logger(__name__)

# The metadata flag that says whether a tracking system was recorded (Head has none)
_ENABLED_KEYS = {
    TrackingSystem.EYES: "eyes_enabled",
    TrackingSystem.HANDS: "hands_enabled",
    TrackingSystem.FACE: "face_enabled",
    TrackingSystem.BODY: "body_enabled",
    TrackingSystem.CONTROLLERS: "controllers_enabled",
}


def is_system_enabled(system: TrackingSystem, metadata: SessionMetadata) -> bool:
    """
    Check if a tracking system was enabled during recording.

    Parameters
    ----------
    system : TrackingSystem
        Tracking system to check
    metadata : SessionMetadata
        Session metadata with feature flags

    Returns
    -------
    bool
        True if system was enabled
    """
    key = _ENABLED_KEYS.get(system)  # Head has no flag: always present
    return True if key is None else getattr(metadata, key)


def systems_off_on_purpose(
    metadata: SessionMetadata, enabled_systems: dict[str, bool] | None
) -> set[TrackingSystem]:
    """Tracking systems turned off on purpose.

    That is, disabled in the config (``systems.<name>.enabled: false``) or set to
    JSON false in the metadata. A missing key or any other value (1/0, a string,
    null) is not on purpose: that stream was expected.
    """
    return {
        system
        for system in TrackingSystem
        if (enabled_systems is not None and not enabled_systems.get(system.value, True))
        or _ENABLED_KEYS.get(system) in metadata.false_flags
    }


def _log_not_enabled_in_metadata(
    session: Session, system: TrackingSystem, consequence: str
) -> None:
    """Explain a stream with data that the metadata does not enable.

    WARNING when the flag was missing or not true/false (a departure from the
    input contract), INFO when the metadata says false.
    """
    key = _ENABLED_KEYS[system]
    problem = session.metadata.flag_problems.get(key)
    if problem:
        logger.warning(
            f"Session '{session.session_id}': {system.value} not exported: metadata does not "
            f"mark it enabled ({problem}); {consequence}"
        )
    else:
        logger.info(
            f"Session '{session.session_id}': {system.value} not exported: {key} is false in "
            f"the metadata; {consequence}"
        )


def _report_unrouted_columns(
    sid: str, all_columns: list[str], alternate_time_columns: dict[str, str] | None
) -> None:
    """Report continuous-data columns that no stream takes.

    One WARNING for columns with no tracking-system prefix and for Face-prefixed
    columns (Face is read only from the face file); INFO for the template's
    ``Custom_`` transform columns. All of them stay in sourcedata/.
    """
    face_prefixes = tuple(SYSTEM_COLUMN_PREFIXES[TrackingSystem.FACE])
    stream_prefixes = tuple(
        prefix
        for system, prefixes in SYSTEM_COLUMN_PREFIXES.items()
        if system != TrackingSystem.FACE
        for prefix in prefixes
    )
    clocks = set((alternate_time_columns or {}).values())
    unknown, face, custom = [], [], []
    for column in all_columns:
        if column == "timestamp" or column in clocks or column.startswith(stream_prefixes):
            continue
        if column.startswith("Custom_"):
            custom.append(column)
        elif column.startswith(face_prefixes):
            face.append(column)
        else:
            unknown.append(column)

    if unknown or face:
        parts = []
        if unknown:
            parts.append(f"{len(unknown)} match no tracking-system prefix ({preview(unknown)})")
        if face:
            parts.append(
                f"{len(face)} have a Face prefix, but Face is read only from the face file "
                f"({preview(face)})"
            )
        logger.warning(
            f"Session '{sid}': {len(unknown) + len(face)} column(s) of the continuous data are "
            f"not exported to BIDS (they stay in sourcedata/): {'; '.join(parts)}"
        )
    if custom:
        logger.info(
            f"Session '{sid}': {len(custom)} Custom_ column(s) (custom transforms) are not "
            f"exported to BIDS motion files; they stay in sourcedata/ ({preview(custom)})"
        )


def _warn_clock_column_missing(sid: str, system: str, column: str, source: str) -> None:
    """A configured per-system clock column is absent, so the stream uses timeSinceStartup."""
    logger.warning(
        f"Session '{sid}': clock column '{column}' for {system} "
        f"(preprocessing.alternate_time_columns) not found in the {source}; {system} is "
        "timed by timeSinceStartup alone, and the checks that need a separate "
        "timeSinceStartup column will not run on it"
    )


def split_continuous_data(
    session: Session,
    enabled_systems: dict[str, bool] | None,
    sampling_frequencies: dict[str, float],
    alternate_time_columns: dict[str, str] | None = None,
) -> dict[TrackingSystem, TrackingStream]:
    """
    Split the monolithic ContinuousData DataFrame into separate TrackingStreams.

    Parameters
    ----------
    session : Session
        Session with raw_continuous_data loaded
    enabled_systems : Dict[str, bool] | None
        Optional dict specifying which systems to include (from config)
    sampling_frequencies : Dict[str, float]
        Expected sampling frequencies for each system (from config)
    alternate_time_columns : Dict[str, str] | None
        Optional mapping of system names to exactly one alternate time column name.
        Example: {"Hands": "Node_HandLeft_Time", "Eyes": "Eyes_Time"}

    Returns
    -------
    Dict[TrackingSystem, TrackingStream]
        Mapping of tracking systems to their data streams
    """
    sid = session.session_id
    df = session.raw_continuous_data
    if df is None or df.empty:
        logger.warning(f"No continuous data to split (session '{sid}')")
        return {}

    all_columns = df.columns.tolist()
    streams: dict[TrackingSystem, TrackingStream] = {}

    for system in TrackingSystem:
        # Skip Face - comes from separate file
        if system == TrackingSystem.FACE:
            continue

        # Get columns for this system
        cols = get_columns_for_system(
            all_columns, system, alternate_time_columns=alternate_time_columns
        )
        n_data = len(cols) - 1

        # Check if system is enabled in config
        if enabled_systems is not None and not enabled_systems.get(system.value, True):
            if n_data:
                logger.info(
                    f"Session '{sid}': {system.value} not exported: disabled in the config "
                    f"(systems.{system.value}.enabled); its {n_data} columns stay only in "
                    "sourcedata/"
                )
            else:
                logger.debug(f"Skipping {system.value} (disabled in config)")
            continue

        # Check if system was enabled during recording
        if not is_system_enabled(system, session.metadata):
            if n_data:
                _log_not_enabled_in_metadata(
                    session, system, f"its {n_data} columns stay only in sourcedata/"
                )
            else:
                logger.debug(f"Skipping {system.value} (not enabled during recording)")
            continue

        # Skip if only timestamp column
        if n_data < 1:
            if system == TrackingSystem.HEAD:
                logger.warning(
                    f"Session '{sid}': no Head columns found in the continuous data; "
                    "Head not exported"
                )
            else:
                logger.warning(
                    f"Session '{sid}': {system.value} enabled in metadata but no "
                    f"{system.value} columns found; {system.value} not exported"
                )
            continue

        clock = (alternate_time_columns or {}).get(system.value)
        if clock and clock not in all_columns:
            _warn_clock_column_missing(sid, system.value, clock, "continuous data")

        # Extract system data
        system_df = df[cols].copy()

        # Rename alternate time column to 'timestamp' for consistency (if not already timestamp)
        if cols[0] != "timestamp":
            alt_time_col = cols[0]
            if alt_time_col in system_df.columns:
                if "timestamp" in system_df.columns:
                    system_df = system_df.rename(columns={"timestamp": GLOBAL_CLOCK_COLUMN})
                system_df = system_df.rename(columns={alt_time_col: "timestamp"})
                logger.debug(f"Using {alt_time_col} as timestamp for {system.value}")
                check_stream_clock(system_df["timestamp"], sid, system.value, alt_time_col)

        # Determine data (non-time) columns
        data_cols = [c for c in system_df.columns if c not in ("timestamp", GLOBAL_CLOCK_COLUMN)]

        # Get sampling frequency from config
        if system.value not in sampling_frequencies:
            raise ConfigurationError(
                f"Tracking system '{system.value}' is enabled but has no entry in "
                "sampling_frequencies. Add it to the configuration file."
            )
        sampling_freq = sampling_frequencies[system.value]

        stream = TrackingStream(
            system=system,
            data=system_df,
            channel_count=len(data_cols),
            sampling_frequency=sampling_freq,
        )

        streams[system] = stream
        logger.info(f"Split {system.value}: {len(system_df)} rows, {len(data_cols)} channels")

    _report_unrouted_columns(sid, all_columns, alternate_time_columns)

    # Face comes from its own file
    face = session.raw_face_data
    face_in_config = enabled_systems is None or enabled_systems.get("Face", True)
    if not face_in_config:
        if face is not None:
            logger.info(
                f"Session '{sid}': Face not exported: disabled in the config "
                "(systems.Face.enabled); the face file stays only in sourcedata/"
            )
    elif not session.metadata.face_enabled:
        if face is not None:
            _log_not_enabled_in_metadata(
                session, TrackingSystem.FACE, "the face file stays only in sourcedata/"
            )
    elif face is None:
        logger.warning(
            f"Session '{sid}': Face not exported: face_enabled is true but no face file was "
            "loaded (none matches input.face_data_pattern, or it could not be read); the "
            "eyes_closed check needs the Face stream and will not run"
        )
    elif face.empty:
        logger.warning(f"Session '{sid}': Face not exported: the face file has no data rows")
    else:
        face_df = face.copy()
        clock = (alternate_time_columns or {}).get("Face")
        if clock and clock not in face_df.columns:
            _warn_clock_column_missing(sid, "Face", clock, "face file")
        if clock and clock in face_df.columns:
            if "timestamp" in face_df.columns:
                face_df = face_df.rename(columns={"timestamp": GLOBAL_CLOCK_COLUMN})
            face_df = face_df.rename(columns={clock: "timestamp"})
            logger.debug(f"Using {clock} as timestamp for Face")
            check_stream_clock(face_df["timestamp"], sid, "Face", clock)

        data_cols = [c for c in face_df.columns if c not in ("timestamp", GLOBAL_CLOCK_COLUMN)]

        if "Face" not in sampling_frequencies:
            raise ConfigurationError(
                "Face tracking is enabled but has no entry in sampling_frequencies. "
                "Add it to the configuration file."
            )
        streams[TrackingSystem.FACE] = TrackingStream(
            system=TrackingSystem.FACE,
            data=face_df,
            channel_count=len(data_cols),
            sampling_frequency=sampling_frequencies["Face"],
        )
        logger.info(f"Added Face: {len(face_df)} rows, {len(data_cols)} channels")

    return streams
