"""
Stream preprocessing for ResXR pipeline.

Provides two independent preprocessing stages:

1. **Quality flag masking** (``apply_quality_masking`` / ``preprocess_stream``):
   NaN-replaces data columns for time segments flagged during validation.
   Preserves temporal continuity (no row deletion). Only applied to the
   derivative dataset; RAW output is never modified. Time columns
   (``timestamp``, ``timeSinceStartup``) are never masked.

2. **BIDS output preparation** (``prepare_motion_data``):
   Converts internal time columns into BIDS-compliant LATENCY channels
   and strips the originals from the output DataFrame:

   - ``timestamp`` → ``latency``: per-system seconds from recording onset
     (first non-zero timestamp via ``find_recording_onset_index``).
   - ``timeSinceStartup`` → ``latency_global``: global Unity clock seconds
     from recording onset.  Only present when the stream uses an alternate
     per-system time column (configured in ``alternate_time_columns``).
   - Pre-onset rows (where the raw timestamp was 0) are set to ``np.nan``
     because no valid timing exists yet.

   Called at write time in ``pipeline.write_bids_output`` for both RAW and
   DERIVATIVE datasets.  The returned DataFrame is passed directly to
   ``write_motion_tsv``, ``generate_channels_tsv``, and
   ``generate_motion_json`` so all three outputs stay consistent.

Pipeline data flow::

    splitter (split only)
        → validate (quality flags use per-system ``timestamp``)
        → preprocess_stream (optional masking for derivative)
        → prepare_motion_data (LATENCY channels + strip internal cols)
        → write BIDS (motion.tsv, channels.tsv, motion.json)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..core.session import TrackingStream
from ..utils import find_recording_offset_index, find_recording_onset_index

logger = get_logger(__name__)


def apply_quality_masking(
    stream: TrackingStream,
    masking_checks: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply quality flag masking via column-specific NaN replacement.

    Preserves temporal continuity by masking values with np.nan rather than
    deleting rows (BIDS-compliant). The BIDS writer converts np.nan to the
    configured missing_values representation.

    Parameters
    ----------
    stream : TrackingStream
        Stream with quality flags to apply
    masking_checks : list[str] | None
        Optional list of check names to apply masking for.
        If None, apply all flags where mask=True.
        If provided, only apply flags from these specific checks.
        Example: ["tracking_loss"] to only apply tracking_loss flags

    Returns
    -------
    pd.DataFrame
        Data with NaN masks applied

    Raises
    ------
    ValueError
        If DataFrame doesn't contain 'timestamp' column
    """
    # Validate input
    if stream.data.empty:
        logger.warning(f"{stream.system.value}: Empty DataFrame, skipping masking")
        return stream.data.copy()

    if "timestamp" not in stream.data.columns:
        raise ValueError(f"{stream.system.value}: DataFrame must contain 'timestamp' column")

    data = stream.data.copy()

    # Get flags where mask=True
    mask_flags = [f for f in stream.quality_flags if f.mask]

    # Select by check name if specified
    if masking_checks is not None:
        mask_flags = [f for f in mask_flags if f.check_name in masking_checks]

    if not mask_flags:
        logger.debug(f"{stream.system.value}: No exclusion flags to apply")
        return data

    # Get data columns (exclude time columns - NEVER mask timestamps)
    time_cols = {"timestamp", "timeSinceStartup"}
    data_cols = [c for c in data.columns if c not in time_cols]

    # Optimize: Use NumPy arrays for vectorized operations
    timestamps = data["timestamp"].values
    data_start, data_end = timestamps.min(), timestamps.max()

    # Build column → mask mapping
    column_masks = {}

    for flag in mask_flags:
        # Skip flags completely outside data range
        if flag.end_time < data_start or flag.start_time > data_end:
            continue

        # Vectorized time mask
        time_mask = (timestamps >= flag.start_time) & (timestamps <= flag.end_time)

        # Determine which columns to mask
        if not flag.target_columns:
            # Empty list = mask ALL columns (except timestamp)
            cols_to_mask = data_cols
        else:
            # Non-empty list = mask only specified columns (excluding timestamp)
            cols_to_mask = [
                c for c in flag.target_columns if c in data.columns and c != "timestamp"
            ]

        # Accumulate masks (union approach: mask if ANY flag marks column)
        for col in cols_to_mask:
            if col not in column_masks:
                column_masks[col] = time_mask.copy()
            else:
                column_masks[col] = column_masks[col] | time_mask

    masked_count = 0
    for col, mask in column_masks.items():
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = data[col].astype(pd.Int64Dtype())
        elif pd.api.types.is_bool_dtype(data[col]):
            data[col] = data[col].astype(pd.BooleanDtype())
        data.loc[mask, col] = pd.NA
        masked_count += mask.sum()

    # Log statistics
    total_values = len(data) * len(column_masks)
    pct = (masked_count / total_values * 100) if total_values > 0 else 0.0
    logger.info(
        f"{stream.system.value}: Masked {masked_count}/{total_values} values "
        f"({pct:.1f}%) across {len(column_masks)} columns from {len(mask_flags)} flags"
    )

    # Warn if too much data is flagged
    if total_values > 0 and pct > 80:
        logger.warning(
            f"{stream.system.value}: {pct:.1f}% of data flagged. "
            "Review quality flags or data collection."
        )

    return data


def preprocess_stream(
    stream: TrackingStream,
    apply_masking: bool = False,
    masking_checks: list[str] | None = None,
) -> None:
    """
    Apply preprocessing steps to a stream.

    Modifies stream.clean_data with optional masking.

    Parameters
    ----------
    stream : TrackingStream
        Stream to preprocess
    apply_masking : bool, optional
        If True, apply quality flag masking via NaN replacement (default: False)
    masking_checks : list[str] | None, optional
        Optional list of check names to mask. If None, apply all (default: None)
    """
    result = stream.data.copy()

    # Apply quality masking (if enabled)
    if apply_masking:
        result = apply_quality_masking(stream, masking_checks)

    stream.clean_data = result


_INTERNAL_TIME_COLS = {"timestamp", "timeSinceStartup"}


def prepare_motion_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a stream DataFrame for BIDS motion output.

    Computes LATENCY channels from internal time columns and removes
    those internal columns.  The result is ready for writing to
    ``motion.tsv`` and for generating ``channels.tsv`` descriptors.

    LATENCY channels follow the BIDS-motion specification:

    - ``latency``: per-system seconds from recording onset (derived from
      the stream's ``timestamp`` column, which may be an alternate
      per-system clock like ``Node_HandLeft_Time``).
    - ``latency_global``: global Unity clock seconds from recording onset
      (derived from ``timeSinceStartup``).  Only added when
      ``timeSinceStartup`` is present in the DataFrame (i.e., when the
      stream uses an alternate time column and the original global
      ``timestamp`` was preserved as ``timeSinceStartup`` by the splitter).

    Recording onset is determined by ``find_recording_onset_index``
    (first finite non-zero value).  Rows before onset are set to
    ``np.nan`` since no valid timing exists yet.

    Parameters
    ----------
    df : pd.DataFrame
        Stream data containing ``timestamp`` and optionally
        ``timeSinceStartup`` (the original global Unity clock).

    Returns
    -------
    pd.DataFrame
        BIDS-ready DataFrame with ``latency`` (and optionally
        ``latency_global``) as the leading columns, and no internal
        time columns (``timestamp``, ``timeSinceStartup``).
    """
    out = df.copy()

    if "timestamp" in out.columns and len(out) > 0:
        ts_vals = out["timestamp"].values
        onset_idx = find_recording_onset_index(ts_vals)
        offset_idx = find_recording_offset_index(ts_vals)
        onset = float(ts_vals[onset_idx]) if onset_idx is not None else 0.0

        latency = out["timestamp"] - onset
        if onset_idx is not None and onset_idx > 0:
            latency.iloc[:onset_idx] = np.nan
        if offset_idx is not None and offset_idx < len(latency) - 1:
            latency.iloc[offset_idx + 1 :] = np.nan

        out.insert(0, "latency", latency)
        logger.debug(f"Added latency channel (onset: {onset:.3f}s)")

    if "timeSinceStartup" in out.columns and len(out) > 0:
        tsu_vals = out["timeSinceStartup"].values
        global_onset_idx = find_recording_onset_index(tsu_vals)
        global_offset_idx = find_recording_offset_index(tsu_vals)
        global_onset = float(tsu_vals[global_onset_idx]) if global_onset_idx is not None else 0.0

        latency_global = out["timeSinceStartup"] - global_onset
        if global_onset_idx is not None and global_onset_idx > 0:
            latency_global.iloc[:global_onset_idx] = np.nan
        if global_offset_idx is not None and global_offset_idx < len(latency_global) - 1:
            latency_global.iloc[global_offset_idx + 1 :] = np.nan

        # Insert right after latency if it exists, otherwise at position 0
        idx = out.columns.get_loc("latency") + 1 if "latency" in out.columns else 0
        out.insert(idx, "latency_global", latency_global)
        logger.debug(f"Added latency_global channel (global onset: {global_onset:.3f}s)")

    # Remove internal time columns
    out = out.drop(columns=[c for c in _INTERNAL_TIME_COLS if c in out.columns])

    return out
