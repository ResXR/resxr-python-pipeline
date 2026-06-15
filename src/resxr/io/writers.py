"""
Output file writers for the ResXR pipeline.

Handles writing BIDS-compliant TSV and JSON files.
"""

from __future__ import annotations

import json
import shutil
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
            lineterminator="\n",  # OS-independent line endings (pandas defaults to os.linesep)
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

        with open(path, "w", encoding="utf-8", newline="\n") as f:
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
    sidecar: dict,
) -> None:
    """Write a finished wide events frame + its sidecar. Dumb writer: no
    column assembly, no dropping. Caller supplies both."""
    output_path = Path(output_path)
    if output_path.suffix != ".tsv":
        raise ValueError(f"events output_path must end in .tsv, got: {output_path}")

    required = ["onset", "duration", "name"]
    missing = [c for c in required if c not in events_df.columns]
    if missing:
        raise BIDSWriteError(f"Events DataFrame missing required columns: {missing}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(output_path, sep="\t", index=False, na_rep="n/a", lineterminator="\n")
        logger.debug(f"Wrote events.tsv: {output_path}")
    except Exception as e:
        raise BIDSWriteError(f"Failed to write events TSV {output_path}: {e}") from e

    json_path = output_path.with_suffix(".json")
    try:
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.debug(f"Wrote events.json: {json_path}")
    except Exception as e:
        raise BIDSWriteError(f"Failed to write events JSON sidecar: {e}") from e


def copy_sourcedata(session_dir: Path, dest_dir: Path, overwrite: bool = False) -> None:
    """Copy a raw session directory verbatim into sourcedata/.

    Verbatim only — no transformation, no filtering. If dest exists and is
    non-empty, skip (and warn) unless overwrite=True, so a re-run never clobbers
    a previously verified copy.
    """
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()) and not overwrite:
        logger.warning(
            "sourcedata dest %s already exists and is non-empty; skipping copy "
            "(set output.overwrite=true to re-copy).",
            dest_dir,
        )
        return
    shutil.copytree(session_dir, dest_dir, dirs_exist_ok=True)
    logger.debug("Copied sourcedata: %s -> %s", session_dir, dest_dir)
