"""Tests for channels.tsv generation (bids/channels.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resxr.bids.channels import detect_hand_joint_frames, generate_channels_tsv
from resxr.io.readers import load_continuous_data

_MUSEUM_CSV = (
    Path(__file__).resolve().parents[2]
    / "DATA"
    / "Demo_Data"
    / "Museum"
    / "2026.06.10_18-42"
    / "2026.06.10_18-42_ContinuousData.csv"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _left_hand_df(wrist_frame: str, n: int = 20) -> pd.DataFrame:
    """Left hand root and Wrist joint, with the Wrist written in *wrist_frame*."""
    rng = np.random.default_rng(3)
    pos = rng.normal(0.0, 0.3, (n, 3))
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    if wrist_frame == "global":
        wrist_pos, wrist_quat = pos, quat
    elif wrist_frame == "tracking_space":  # also negates the whole quaternion (same rotation)
        wrist_pos, wrist_quat = pos * [1, 1, -1], quat * [1, 1, -1, -1]
    elif wrist_frame == "neither":
        wrist_pos, wrist_quat = pos + 0.05, quat
    else:  # "no usable rows": positions all zero
        pos, wrist_pos, wrist_quat = pos * 0, pos * 0, quat
    data = {}
    for i, c in enumerate(("px", "py", "pz", "qx", "qy", "qz", "qw")):
        data[f"LeftHand_Root_{c}"] = np.hstack([pos, quat])[:, i]
    for i, c in enumerate(("x", "y", "z", "qx", "qy", "qz", "qw")):
        data[f"Left_XRHand_Wrist_{c}"] = np.hstack([wrist_pos, wrist_quat])[:, i]
    return pd.DataFrame(data)


def _prepared_head_df() -> pd.DataFrame:
    """Minimal BIDS-ready motion DataFrame for a HEAD stream."""
    n = 10
    return pd.DataFrame(
        {
            "latency": np.linspace(0.0, 1.0, n),
            "Node_Head_px": np.zeros(n),
            "Node_Head_py": np.zeros(n),
            "Node_Head_pz": np.zeros(n),
            "Node_Head_qx": np.zeros(n),
            "Node_Head_qy": np.zeros(n),
            "Node_Head_qz": np.zeros(n),
            "Node_Head_qw": np.ones(n),
        }
    )


# ===========================================================================
# generate_channels_tsv
# ===========================================================================


class TestGenerateChannelsTsv:
    def test_returns_dataframe(self):
        """Returns a pandas DataFrame."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        """Result has all required BIDS channels.tsv columns."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        required = {
            "name",
            "component",
            "type",
            "tracked_point",
            "units",
            "sampling_frequency",
            "reference_frame",
        }
        assert required.issubset(set(result.columns))

    def test_one_row_per_input_column(self):
        """Number of rows equals number of columns in the input DataFrame."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        assert len(result) == len(df.columns)

    def test_name_column_matches_input_columns(self):
        """'name' column contains the same strings as df.columns (in order)."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        assert list(result["name"]) == list(df.columns)

    def test_type_column_has_no_nulls(self):
        """'type' column contains no None or NaN values."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert result["type"].notna().all()

    def test_units_column_has_no_nulls(self):
        """'units' column contains no None or NaN values."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert result["units"].notna().all()

    def test_sampling_frequency_column_value(self):
        """sampling_frequency column contains the provided frequency for all rows."""
        result = generate_channels_tsv(_prepared_head_df(), 90.0)
        assert (result["sampling_frequency"] == 90.0).all()

    def test_empty_dataframe_returns_empty(self):
        """Empty input DataFrame → 0-row result with correct columns."""
        result = generate_channels_tsv(pd.DataFrame(), 90.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "col,expected_type",
        [
            ("Node_Head_px", "POS"),
            ("Node_Head_py", "POS"),
            ("Node_Head_qx", "ORNT"),
            ("Node_Head_qw", "ORNT"),
            ("latency", "LATENCY"),
        ],
    )
    def test_known_column_type(self, col, expected_type):
        """Each known column has the expected BIDS channel type."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        row = result[result["name"] == col]
        assert len(row) == 1
        assert row.iloc[0]["type"] == expected_type

    def test_latency_tracked_point_is_na(self):
        """LATENCY channels have tracked_point = 'n/a'."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        latency_row = result[result["name"] == "latency"].iloc[0]
        assert latency_row["tracked_point"] == "n/a"

    def test_spatial_columns_have_global_reference_frame(self):
        """POS and ORNT columns have reference_frame = 'global'."""
        df = _prepared_head_df()
        result = generate_channels_tsv(df, 90.0)
        pos_rows = result[result["type"] == "POS"]
        assert (pos_rows["reference_frame"] == "global").all()

    def test_misc_columns_have_na_reference_frame(self):
        """MISC columns have reference_frame = 'n/a'."""
        df = pd.DataFrame(
            {
                "LeftHand_Status_HandTracked": [1, 0],
            }
        )
        result = generate_channels_tsv(df, 90.0)
        misc_rows = result[result["type"] == "MISC"]
        if len(misc_rows) > 0:
            assert (misc_rows["reference_frame"] == "n/a").all()


# ===========================================================================
# detect_hand_joint_frames and the joint channels' reference_frame
# ===========================================================================


class TestHandJointFrames:
    @pytest.mark.parametrize("frame", ["global", "tracking_space"])
    def test_detects_frame_of_wrist_joint(self, frame):
        """Wrist equal to the root -> global; z and quaternion x, y negated -> tracking_space."""
        assert detect_hand_joint_frames(_left_hand_df(frame)) == {"Left": frame}

    @pytest.mark.parametrize("case", ["neither", "no usable rows"])
    def test_unknown_frame_is_na_with_warning(self, case, caplog):
        """No matching rule, or no usable rows -> n/a and a warning naming session and hand."""
        assert detect_hand_joint_frames(_left_hand_df(case), "sess_x") == {"Left": "n/a"}
        assert "Session 'sess_x': Left hand joint frame unknown" in caplog.text

    def test_joint_channels_get_detected_frame(self):
        """Only XRHand joint POS/ORNT channels take the detected frame; the root stays global."""
        result = generate_channels_tsv(_left_hand_df("global"), 90.0, {"Left": "tracking_space"})
        frames = dict(zip(result["name"], result["reference_frame"], strict=True))
        assert frames["Left_XRHand_Wrist_z"] == "tracking_space"
        assert frames["Left_XRHand_Wrist_qw"] == "tracking_space"
        assert frames["LeftHand_Root_pz"] == "global"

    def test_museum_demo_joints_are_in_tracking_space(self):
        """Both hands' joints in the Museum demo session are in tracking space."""
        if not _MUSEUM_CSV.exists():
            pytest.skip(f"Museum demo CSV not found: {_MUSEUM_CSV}")
        with _MUSEUM_CSV.open("rb") as f:
            if f.read(8) == b"version ":  # Git LFS pointer (checkout without LFS, as in CI)
                pytest.skip(f"Museum demo CSV is a Git LFS pointer: {_MUSEUM_CSV}")
        data = load_continuous_data(_MUSEUM_CSV)
        assert detect_hand_joint_frames(data) == {
            "Left": "tracking_space",
            "Right": "tracking_space",
        }
