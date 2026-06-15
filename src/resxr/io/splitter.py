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

BIDS LATENCY channels are **not** computed here.  That is handled
downstream by ``prepare_motion_data`` in the preprocessing module,
which runs at write time.
"""

from __future__ import annotations

from ..core.constants import GLOBAL_CLOCK_COLUMN, TrackingSystem
from ..core.exceptions import ConfigurationError
from ..core.logger import get_logger
from ..core.session import Session, SessionMetadata, TrackingStream
from .column_maps import get_columns_for_system

logger = get_logger(__name__)


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
    # Head is always present
    if system == TrackingSystem.HEAD:
        return True

    mapping = {
        TrackingSystem.EYES: metadata.eyes_enabled,
        TrackingSystem.HANDS: metadata.hands_enabled,
        TrackingSystem.FACE: metadata.face_enabled,
        TrackingSystem.BODY: metadata.body_enabled,
        TrackingSystem.CONTROLLERS: metadata.controllers_enabled,
    }

    return mapping.get(system, True)


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
    df = session.raw_continuous_data
    if df is None or df.empty:
        logger.warning("No continuous data to split")
        return {}

    all_columns = df.columns.tolist()
    streams: dict[TrackingSystem, TrackingStream] = {}

    for system in TrackingSystem:
        # Skip Face - comes from separate file
        if system == TrackingSystem.FACE:
            continue

        # Check if system is enabled in config
        if enabled_systems is not None and not enabled_systems.get(system.value, True):
            logger.debug(f"Skipping {system.value} (disabled in config)")
            continue

        # Check if system was enabled during recording
        if not is_system_enabled(system, session.metadata):
            logger.debug(f"Skipping {system.value} (not enabled during recording)")
            continue

        # Get columns for this system
        cols = get_columns_for_system(
            all_columns, system, alternate_time_columns=alternate_time_columns
        )

        # Skip if only timestamp column
        if len(cols) <= 1:
            logger.debug(f"Skipping {system.value} (no data columns)")
            continue

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

        # Determine data (non-time) columns
        data_cols = [c for c in system_df.columns if c not in ("timestamp", GLOBAL_CLOCK_COLUMN)]

        # Get sampling frequency from config
        if system.value not in sampling_frequencies:
            raise ConfigurationError(
                f"Tracking system '{system.value}' is enabled but has no entry in "
                "sampling_frequencies. Add it to config/pipeline_config.yaml."
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

    # Handle Face data from separate file
    if (
        session.raw_face_data is not None
        and (enabled_systems is None or enabled_systems.get("Face", True))
        and session.metadata.face_enabled
    ):
        face_df = session.raw_face_data.copy()

        if not face_df.empty:
            # Use alternate time column if configured
            if alternate_time_columns:
                alt_time_col = alternate_time_columns.get("Face")
                if alt_time_col and alt_time_col in face_df.columns:
                    if "timestamp" in face_df.columns:
                        face_df = face_df.rename(columns={"timestamp": GLOBAL_CLOCK_COLUMN})
                    face_df = face_df.rename(columns={alt_time_col: "timestamp"})
                    logger.debug(f"Using {alt_time_col} as timestamp for Face")

            data_cols = [c for c in face_df.columns if c not in ("timestamp", GLOBAL_CLOCK_COLUMN)]

            # Get sampling frequency from config
            if "Face" not in sampling_frequencies:
                raise ConfigurationError(
                    "Face tracking is enabled but has no entry in sampling_frequencies. "
                    "Add it to config/pipeline_config.yaml."
                )
            sampling_freq = sampling_frequencies["Face"]

            stream = TrackingStream(
                system=TrackingSystem.FACE,
                data=face_df,
                channel_count=len(data_cols),
                sampling_frequency=sampling_freq,
            )
            streams[TrackingSystem.FACE] = stream
            logger.info(f"Added Face: {len(face_df)} rows, {len(data_cols)} channels")

    return streams
