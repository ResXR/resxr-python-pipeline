"""
Output file writers for the ResXR pipeline.

Handles writing BIDS-compliant TSV and JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.exceptions import BIDSWriteError
from ..core.logger import get_logger

logger = get_logger(__name__)


def write_bids_tsv(
    df: pd.DataFrame,
    path: Path,
    include_header: bool = True,
    float_format: str | None = None,
    missing_values: str = "n/a",
) -> None:
    """
    Write DataFrame to BIDS-compliant TSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    path : Path
        Output file path
    include_header : bool
        Whether to include column headers (False for motion.tsv)
    float_format : str | None
        Format string for float values.  ``None`` (default) preserves
        full float64 precision.
    missing_values : str
        String representation for missing/NaN values (from config)

    Raises
    ------
    BIDSWriteError
        If file cannot be written
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            path,
            sep="\t",
            index=False,
            header=include_header,
            float_format=float_format,
            na_rep=missing_values,
        )
        logger.debug(f"Wrote TSV: {path}")

    except Exception as e:
        raise BIDSWriteError(f"Failed to write {path}: {e}") from e


def write_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    """
    Write dictionary to JSON file with consistent formatting.

    Parameters
    ----------
    data : Dict[str, Any]
        Data to write
    path : Path
        Output file path
    indent : int
        JSON indentation level

    Raises
    ------
    BIDSWriteError
        If file cannot be written
    """
    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.write("\n")  # Trailing newline
        logger.debug(f"Wrote JSON: {path}")

    except Exception as e:
        raise BIDSWriteError(f"Failed to write {path}: {e}") from e


def write_motion_tsv(
    df: pd.DataFrame,
    path: Path,
    missing_values: str,
    float_format: str | None = None,
) -> None:
    """
    Write motion data TSV (no header, tab-separated).

    BIDS motion.tsv files have no header row — columns are described
    in the accompanying channels.tsv file.

    The DataFrame must already be BIDS-ready (output of
    ``prepare_motion_data``): it should contain ``latency`` /
    ``latency_global`` LATENCY channels and no internal time columns
    (``timestamp``, ``timeSinceStartup``).

    Parameters
    ----------
    df : pd.DataFrame
        Motion data (already prepared via ``prepare_motion_data``)
    path : Path
        Output file path
    missing_values : str
        String representation for missing/NaN values (from config)
    float_format : str | None
        Format string for float values.  ``None`` (default) preserves
        full float64 precision.
    """
    write_bids_tsv(
        df,
        path,
        include_header=False,
        float_format=float_format,
        missing_values=missing_values,
    )


def write_channels_tsv(df: pd.DataFrame, path: Path) -> None:
    """
    Write channels descriptor TSV (with header).

    Parameters
    ----------
    df : pd.DataFrame
        Channels descriptor data
    path : Path
        Output file path
    """
    write_bids_tsv(df, path, include_header=True)


def write_participants_tsv(df: pd.DataFrame, path: Path) -> None:
    """
    Write participants.tsv file.

    Parameters
    ----------
    df : pd.DataFrame
        Participant information
    path : Path
        Output file path
    """
    write_bids_tsv(df, path, include_header=True)


def write_scans_tsv(df: pd.DataFrame, path: Path) -> None:
    """
    Write scans.tsv file for a session.

    Parameters
    ----------
    df : pd.DataFrame
        Scans information with filename and acq_time columns
    path : Path
        Output file path
    """
    write_bids_tsv(df, path, include_header=True)


def write_bids_events(
    events_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write events data to BIDS-compliant events.tsv file.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events data with columns: onset, duration, name
    output_path : Path
        Path to output events.tsv file

    Raises
    ------
    BIDSWriteError
        If file or JSON sidecar cannot be written or validation fails
    """
    output_path = Path(output_path)

    required_cols = ["onset", "duration", "name"]
    missing = [c for c in required_cols if c not in events_df.columns]
    if missing:
        raise BIDSWriteError(f"Events DataFrame missing required columns: {missing}")

    events_df = events_df.copy()
    events_df = events_df.sort_values("onset")

    extra_cols = [c for c in events_df.columns if c not in required_cols]
    if extra_cols:
        logger.warning(
            "Events DataFrame has extra columns that will be dropped: %s",
            extra_cols,
        )
    events_df = events_df[required_cols]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            na_rep="n/a",
        )
        logger.debug(f"Wrote events.tsv: {output_path}")
    except Exception as e:
        raise BIDSWriteError(f"Failed to write events TSV {output_path}: {e}") from e

    # Write JSON sidecar
    events_json_path = output_path.with_suffix(".json")
    events_metadata = {
        "onset": {
            "Description": "Onset time of event in seconds relative to recording start",
            "Units": "s",
        },
        "duration": {
            "Description": "Duration of event in seconds (0 for instantaneous events)",
            "Units": "s",
        },
        "name": {
            "Description": "Type or name of the event",
            "LongName": "Event Type",
        },
    }

    try:
        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump(events_metadata, f, indent=2)
        logger.debug(f"Wrote events.json: {events_json_path}")
    except Exception as e:
        raise BIDSWriteError(f"Failed to write events JSON sidecar: {e}") from e
