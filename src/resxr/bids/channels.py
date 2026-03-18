"""
BIDS channels.tsv generation for ResXR pipeline.

Generates channel descriptor files documenting column structure.

Accepts a **prepared** DataFrame (output of ``prepare_motion_data``)
that already contains BIDS LATENCY channels (``latency``,
``latency_global``) and no internal time columns.  Each column in the
DataFrame becomes one row in ``channels.tsv``.
"""

from __future__ import annotations

import pandas as pd

from ..io.column_maps import extract_tracked_point, infer_bids_channel_info


def generate_channels_tsv(
    data: pd.DataFrame,
    sampling_frequency: float,
) -> pd.DataFrame:
    """
    Generate channels.tsv descriptor for prepared motion data.

    The channels.tsv file documents each column in the motion.tsv file,
    including its type, component, units, and tracked point.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared motion data (output of prepare_motion_data, no internal
        time columns)
    sampling_frequency : float
        Nominal sampling frequency in Hz

    Returns
    -------
    pd.DataFrame
        Channels descriptor with BIDS-required columns
    """
    rows = []

    for col in data.columns:
        # Infer channel metadata
        ctype, component, units = infer_bids_channel_info(col)
        tracked_point = extract_tracked_point(col)

        # LATENCY channels are not spatial tracked points
        if ctype == "LATENCY":
            tracked_point = "n/a"

        # Determine reference frame (spatial data uses global, others n/a)
        spatial_types = {"POS", "ORNT", "VEL", "GYRO", "ACCEL", "ANGACCEL"}
        ref_frame = "global" if ctype in spatial_types else "n/a"

        rows.append(
            {
                "name": col,
                "component": component,
                "type": ctype,
                "tracked_point": tracked_point,
                "units": units,
                "sampling_frequency": sampling_frequency,
                "reference_frame": ref_frame,
            }
        )

    return pd.DataFrame(rows)
