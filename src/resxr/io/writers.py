"""
Output file writers for the ResXR pipeline.

Writes BIDS TSV and JSON files, copies a session into sourcedata/, and
deletes output (``remove_path``), including files marked read-only.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
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


def _clear_read_only(path: Path) -> None:
    """Clear the read-only attribute (Windows) or add the owner's write bit."""
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)


def _retry_read_only(func, path, exc: BaseException) -> None:
    """rmtree onexc handler: clear a read-only file or folder and delete it again."""
    if func not in (os.unlink, os.remove, os.rmdir):
        raise exc
    _clear_read_only(Path(path))
    func(path)


def _retry_read_only_legacy(func, path, exc_info) -> None:
    """rmtree onerror handler (Python < 3.12): the same, from an exc_info tuple."""
    _retry_read_only(func, path, exc_info[1])


def remove_path(path: Path) -> None:
    """Delete a file or a folder tree, including files marked read-only.

    Raises OSError when a file or folder still cannot be deleted.
    """
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry_read_only)
        else:
            shutil.rmtree(path, onerror=_retry_read_only_legacy)
        return
    try:
        path.unlink()
    except PermissionError:
        _clear_read_only(path)
        path.unlink()


def _is_input(session_dir: Path, dest_dir: Path) -> bool:
    """Whether *dest_dir* is the session's input folder itself."""
    return Path(dest_dir).resolve() == Path(session_dir).resolve()


def will_copy_sourcedata(session_dir: Path, dest_dir: Path, overwrite: bool) -> bool:
    """Whether ``copy_sourcedata`` copies into *dest_dir*.

    It does not when *dest_dir* is the input folder itself, nor into a
    non-empty *dest_dir* without *overwrite*.
    """
    dest_dir = Path(dest_dir)
    if _is_input(session_dir, dest_dir):
        return False
    return overwrite or not (dest_dir.exists() and any(dest_dir.iterdir()))


def copy_sourcedata(session_dir: Path, dest_dir: Path, overwrite: bool = False) -> None:
    """Copy a raw session directory verbatim into sourcedata/.

    Verbatim only: no transformation, no filtering. If dest exists and is
    non-empty, skip (and warn) unless overwrite=True, so a re-run never clobbers
    a previously verified copy. With overwrite, the input is copied over the
    existing copy, after its read-only files are made writable; a file no longer
    in the input stays. When *dest_dir* is the input folder itself (the session
    is read from sourcedata/), nothing is copied.
    """
    dest_dir = Path(dest_dir)
    if _is_input(session_dir, dest_dir):
        logger.info(f"sourcedata is the input folder; nothing copied ({dest_dir})")
        return
    if not will_copy_sourcedata(session_dir, dest_dir, overwrite):
        logger.warning(
            "sourcedata dest %s already exists and is non-empty; skipping copy "
            "(set output.overwrite=true to copy the input over it).",
            dest_dir,
        )
        return
    if dest_dir.exists():
        for path in dest_dir.rglob("*"):
            if path.is_file() and not os.access(path, os.W_OK):
                _clear_read_only(path)
    shutil.copytree(session_dir, dest_dir, dirs_exist_ok=True)
    logger.debug("Copied sourcedata: %s -> %s", session_dir, dest_dir)
