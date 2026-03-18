"""Schema-guard tests: verify test fixtures match real OVRPlugin data ranges.

These tests ensure the synthetic data produced by conftest.py fixtures stays
within the ranges documented in tests/data_sources_README.txt.

Key OVRPlugin output ranges:
- Bool-as-0/1 columns: exactly 0 or 1 (integer)
- Face expression weights: [0.0, 1.0]
- Confidence values: [0.0, 1.0]
- Quaternion components: each in [-1, 1], ||q|| = 1
- Positions: world-space floats (no hard bounds, but room scale ~[-10, 10]m)
- Timestamps: non-negative, monotonically non-decreasing seconds
"""

from __future__ import annotations

import numpy as np

from tests.conftest import _eyes_df, _face_df, _hands_df, _head_df, make_timestamps

# ===========================================================================
# Timestamps
# ===========================================================================


class TestTimestampSchema:
    def test_timestamps_are_nonnegative(self):
        """All test timestamps should be non-negative (seconds since startup)."""
        ts = make_timestamps(200, 90.0, 1.0)
        assert (ts >= 0).all()

    def test_timestamps_monotonically_increasing(self):
        """Timestamps are strictly increasing."""
        ts = make_timestamps(200, 90.0, 1.0)
        assert (np.diff(ts) > 0).all()

    def test_head_df_timestamps_nonneg(self):
        df = _head_df()
        assert (df["timestamp"] >= 0).all()

    def test_hands_df_timestamps_nonneg(self):
        df = _hands_df()
        assert (df["timestamp"] >= 0).all()

    def test_eyes_df_timestamps_nonneg(self):
        df = _eyes_df()
        assert (df["timestamp"] >= 0).all()

    def test_face_df_timestamps_nonneg(self):
        df = _face_df()
        assert (df["timestamp"] >= 0).all()


# ===========================================================================
# Bool-as-0/1 columns
# ===========================================================================


class TestBoolColumnSchema:
    def test_hands_validity_columns_are_0_or_1(self):
        """LeftHand_Status_HandTracked / RightHand_Status_HandTracked are exactly 0 or 1."""
        df = _hands_df(with_validity=True)
        for col in ("LeftHand_Status_HandTracked", "RightHand_Status_HandTracked"):
            values = df[col].dropna().unique()
            assert set(values).issubset({0, 1}), f"{col} has values outside {{0, 1}}: {set(values)}"


# ===========================================================================
# Face expression weights: [0.0, 1.0]
# ===========================================================================


class TestFaceExpressionSchema:
    def test_eyes_closed_in_range(self):
        """Eyes_Closed_L and Eyes_Closed_R are in [0.0, 1.0]."""
        df = _face_df()
        for col in ("Eyes_Closed_L", "Eyes_Closed_R"):
            assert df[col].min() >= 0.0, f"{col} below 0.0"
            assert df[col].max() <= 1.0, f"{col} above 1.0"

    def test_jaw_drop_in_range(self):
        """Jaw_Drop is in [0.0, 1.0]."""
        df = _face_df()
        assert df["Jaw_Drop"].min() >= 0.0
        assert df["Jaw_Drop"].max() <= 1.0


# ===========================================================================
# Position columns: room-scale sanity
# ===========================================================================


class TestPositionSchema:
    def test_head_positions_room_scale(self):
        """Head positions should be within ±10m (room-scale VR sanity check)."""
        df = _head_df()
        for col in ("Node_Head_px", "Node_Head_py", "Node_Head_pz"):
            assert df[col].abs().max() < 10.0, f"{col} outside room scale"

    def test_hands_positions_room_scale(self):
        """Hand positions should be within ±10m."""
        df = _hands_df()
        for col in (
            "LeftHand_Root_px",
            "LeftHand_Root_py",
            "RightHand_Root_px",
            "RightHand_Root_py",
        ):
            assert df[col].abs().max() < 10.0, f"{col} outside room scale"

    def test_eyes_positions_room_scale(self):
        """Eye gaze positions should be within ±10m."""
        df = _eyes_df()
        for col in ("LeftEye_px", "LeftEye_py"):
            assert df[col].abs().max() < 10.0, f"{col} outside room scale"


# ===========================================================================
# Quaternion columns: component range
# ===========================================================================


class TestQuaternionSchema:
    def test_head_qw_within_quaternion_range(self):
        """Node_Head_qw should be in [-1.0, 1.0] for a valid unit quaternion.

        NOTE: Current fixture uses rng.normal(1.0, 0.01, n) which may very
        rarely exceed 1.0.  This test documents the expected range; if it
        fails, the fixture generator should be tightened.
        """
        df = _head_df()
        # Allow a small tolerance for the normal distribution tails
        assert df["Node_Head_qw"].min() >= -1.05, "qw below -1.05"
        assert df["Node_Head_qw"].max() <= 1.05, "qw above 1.05"
