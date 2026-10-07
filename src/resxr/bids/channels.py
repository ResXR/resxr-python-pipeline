"""
BIDS channels.tsv generation for ResXR pipeline.

Generates channel descriptor files documenting column structure.

Accepts a **prepared** DataFrame (output of ``prepare_motion_data``)
that already contains BIDS LATENCY channels (``latency``,
``latency_global``) and no internal time columns.  Each column in the
DataFrame becomes one row in ``channels.tsv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.logger import get_logger
from ..io.column_maps import extract_tracked_point, infer_bids_channel_info

logger = get_logger(__name__)

GLOBAL_FRAME = "global"
TRACKING_SPACE_FRAME = "tracking_space"
UNKNOWN_FRAME = "n/a"

_HAND_SIDES = ("Left", "Right")
_POSE_TOLERANCE = 1e-5  # metres for positions, unit-free for quaternion components
_MIN_MATCH_FRACTION = 0.99


def _hand_joint_side(column: str) -> str | None:
    """'Left' or 'Right' for an XRHand joint column, else None."""
    return next((side for side in _HAND_SIDES if column.startswith(f"{side}_XRHand_")), None)


def _detect_hand_joint_frame(data: pd.DataFrame, side: str, session_id: str) -> str:
    """Frame of one hand's XRHand joints, from its Wrist joint vs. its hand root."""
    wrist = [f"{side}_XRHand_Wrist_{c}" for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    root = [f"{side}Hand_Root_{c}" for c in ("px", "py", "pz", "qx", "qy", "qz", "qw")]
    if not set(wrist + root).issubset(data.columns):
        logger.warning(
            f"Session '{session_id}': {side} hand joint frame unknown "
            f"(Wrist joint or hand root columns missing); reference_frame n/a"
        )
        return UNKNOWN_FRAME

    w = data[wrist].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    r = data[root].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    usable = (
        np.isfinite(w).all(axis=1)
        & np.isfinite(r).all(axis=1)
        & (w[:, :3] != 0).any(axis=1)
        & (r[:, :3] != 0).any(axis=1)
    )
    w, r = w[usable], r[usable]

    def share_matching(position_sign: list[int], quaternion_sign: list[int]) -> float:
        expected_pos = r[:, :3] * position_sign
        expected_quat = r[:, 3:] * quaternion_sign
        pos_ok = (np.abs(w[:, :3] - expected_pos) <= _POSE_TOLERANCE).all(axis=1)
        quat_ok = (np.abs(w[:, 3:] - expected_quat) <= _POSE_TOLERANCE).all(axis=1) | (
            np.abs(w[:, 3:] + expected_quat) <= _POSE_TOLERANCE
        ).all(axis=1)  # q and -q are the same rotation
        return float((pos_ok & quat_ok).mean())

    if len(w):
        is_global = share_matching([1, 1, 1], [1, 1, 1, 1]) >= _MIN_MATCH_FRACTION
        is_tracking_space = share_matching([1, 1, -1], [-1, -1, 1, 1]) >= _MIN_MATCH_FRACTION
        if is_global != is_tracking_space:
            return GLOBAL_FRAME if is_global else TRACKING_SPACE_FRAME

    logger.warning(
        f"Session '{session_id}': {side} hand joint frame unknown "
        f"(no single frame fits the Wrist joint and hand root on {len(w)} usable rows); "
        f"reference_frame n/a"
    )
    return UNKNOWN_FRAME


def detect_hand_joint_frames(data: pd.DataFrame, session_id: str = "") -> dict[str, str]:
    """
    Detect the frame of each hand's XRHand joint poses in a session.

    Depending on the Unity template build, the joints are written in the
    same frame as the hand root (the Wrist joint equals the root:
    ``"global"``) or as the SDK's raw tracking-space values (Wrist z and
    the quaternion's x and y have the root's opposite sign:
    ``"tracking_space"``). Rows where either position is missing, not
    finite or all zero are ignored. When neither rule holds on at least
    99% of the remaining rows, or no row remains, the frame is ``"n/a"``
    and a warning is logged.

    Parameters
    ----------
    data : pd.DataFrame
        Hands stream data (joint and hand root columns)
    session_id : str
        Session name used in warnings

    Returns
    -------
    dict[str, str]
        Frame per hand ("Left", "Right"); hands without joint columns are omitted
    """
    sides = {side for col in data.columns if (side := _hand_joint_side(col))}
    return {
        side: _detect_hand_joint_frame(data, side, session_id)
        for side in _HAND_SIDES
        if side in sides
    }


def generate_channels_tsv(
    data: pd.DataFrame,
    sampling_frequency: float,
    hand_joint_frames: dict[str, str] | None = None,
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
    hand_joint_frames : dict[str, str] | None
        Detected frame per hand (``detect_hand_joint_frames``); used as the
        reference_frame of that hand's XRHand joint POS and ORNT channels

    Returns
    -------
    pd.DataFrame
        Channels descriptor with BIDS-required columns
    """
    rows = []
    hand_joint_frames = hand_joint_frames or {}

    for col in data.columns:
        # Infer channel metadata
        ctype, component, units = infer_bids_channel_info(col)
        tracked_point = extract_tracked_point(col)

        # LATENCY channels are not spatial tracked points
        if ctype == "LATENCY":
            tracked_point = "n/a"

        # Determine reference frame (spatial data uses global, others n/a)
        spatial_types = {"POS", "ORNT", "VEL", "GYRO", "ACCEL", "ANGACCEL"}
        ref_frame = GLOBAL_FRAME if ctype in spatial_types else "n/a"

        # XRHand joint poses use the frame detected for their hand
        side = _hand_joint_side(col)
        if ctype in {"POS", "ORNT"} and side in hand_joint_frames:
            ref_frame = hand_joint_frames[side]

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
