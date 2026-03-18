"""Tests for column mapping utilities (io/column_maps.py)."""

from __future__ import annotations

import pytest

from resxr.core.constants import TrackingSystem
from resxr.io.column_maps import (
    classify_columns_by_system,
    count_channel_types,
    count_tracked_points,
    get_columns_for_system,
    infer_bids_channel_info,
)

# ---------------------------------------------------------------------------
# Sample column lists for re-use
# ---------------------------------------------------------------------------

HEAD_COLS = [
    "timestamp",
    "Node_Head_px",
    "Node_Head_py",
    "Node_Head_pz",
    "Node_Head_qx",
    "Node_Head_qy",
    "Node_Head_qz",
    "Node_Head_qw",
]

HANDS_COLS = [
    "timestamp",
    "LeftHand_Root_px",
    "LeftHand_Root_py",
    "RightHand_Root_px",
    "RightHand_Root_py",
    "LeftHand_Status_HandTracked",
    "RightHand_Status_HandTracked",
]

EYES_COLS = [
    "timestamp",
    "LeftEye_px",
    "LeftEye_py",
    "LeftEye_pz",
]

ALL_COLS = HEAD_COLS[1:] + HANDS_COLS[1:] + EYES_COLS[1:] + ["timestamp"]


# ===========================================================================
# get_columns_for_system
# ===========================================================================


class TestGetColumnsForSystem:
    def test_head_columns_returned(self):
        """get_columns_for_system extracts HEAD-prefixed columns."""
        cols = get_columns_for_system(HEAD_COLS, TrackingSystem.HEAD)
        assert "Node_Head_px" in cols
        assert "Node_Head_qw" in cols

    def test_hands_columns_returned(self):
        """get_columns_for_system extracts HANDS-prefixed columns."""
        cols = get_columns_for_system(HANDS_COLS, TrackingSystem.HANDS)
        assert "LeftHand_Root_px" in cols
        assert "RightHand_Root_px" in cols

    def test_timestamp_always_first(self):
        """'timestamp' is included (as the time column) for any system."""
        cols = get_columns_for_system(HEAD_COLS, TrackingSystem.HEAD)
        assert "timestamp" in cols

    def test_no_match_returns_only_timestamp(self):
        """Columns not matching any HEAD prefix result in only timestamp."""
        only_hands = ["timestamp", "LeftHand_Root_px", "RightHand_Root_px"]
        cols = get_columns_for_system(only_hands, TrackingSystem.HEAD)
        # HEAD has no matching data columns here → only timestamp
        assert cols == ["timestamp"]

    def test_eyes_columns_not_mixed_into_head(self):
        """LeftEye_ columns do not appear in HEAD stream columns."""
        mixed = HEAD_COLS + ["LeftEye_px", "LeftEye_py"]
        head_cols = get_columns_for_system(mixed, TrackingSystem.HEAD)
        assert "LeftEye_px" not in head_cols

    def test_alternate_time_column_used_when_configured(self):
        """With alternate_time_columns, the configured column replaces 'timestamp'."""
        cols_with_eyes_time = HEAD_COLS + ["Eyes_Time"]
        result = get_columns_for_system(
            cols_with_eyes_time,
            TrackingSystem.EYES,
            alternate_time_columns={"Eyes": "Eyes_Time"},
        )
        assert "Eyes_Time" in result


# ===========================================================================
# classify_columns_by_system
# ===========================================================================


class TestClassifyColumnsBySystem:
    def test_returns_dict_with_systems(self):
        """Result is a dict keyed by TrackingSystem."""
        result = classify_columns_by_system(ALL_COLS)
        assert isinstance(result, dict)
        for key in result:
            assert isinstance(key, TrackingSystem)

    def test_head_classified_correctly(self):
        """Node_Head_ columns are under HEAD key."""
        result = classify_columns_by_system(ALL_COLS)
        if TrackingSystem.HEAD in result:
            assert "Node_Head_px" in result[TrackingSystem.HEAD]

    def test_no_overlap_between_systems(self):
        """No data column appears in more than one system's column list."""
        result = classify_columns_by_system(ALL_COLS)
        seen = {}
        for system, cols in result.items():
            for col in cols:
                if col == "timestamp":
                    continue  # timestamp is shared
                assert col not in seen, f"{col!r} appears in both {seen.get(col)} and {system}"
                seen[col] = system

    def test_unrecognised_column_not_classified(self):
        """A custom 'UnknownSensor_X' column is not in any system's list."""
        cols = ALL_COLS + ["UnknownSensor_X"]
        result = classify_columns_by_system(cols)
        for _system, system_cols in result.items():
            assert "UnknownSensor_X" not in system_cols


# ===========================================================================
# infer_bids_channel_info
# ===========================================================================


@pytest.mark.parametrize(
    "column,expected_type,expected_component",
    [
        ("Node_Head_px", "POS", "x"),
        ("Node_Head_py", "POS", "y"),
        ("Node_Head_pz", "POS", "z"),
        ("Node_Head_qx", "ORNT", "quat_x"),
        ("Node_Head_qy", "ORNT", "quat_y"),
        ("Node_Head_qz", "ORNT", "quat_z"),
        ("Node_Head_qw", "ORNT", "quat_w"),
        ("Eyes_Time", "LATENCY", "n/a"),
        ("latency", "LATENCY", "n/a"),
        ("latency_global", "LATENCY", "n/a"),
        ("LeftHand_Status_HandTracked", "MISC", "n/a"),
        ("LeftHand_Confidence", "MISC", "n/a"),
        ("LeftEye_px", "POS", "x"),
    ],
)
def test_infer_bids_channel_info_known_columns(column, expected_type, expected_component):
    """Known columns return the expected channel_type and component."""
    ctype, component, units = infer_bids_channel_info(column)
    assert ctype == expected_type
    assert component == expected_component


def test_infer_bids_channel_info_returns_tuple_of_3_strings():
    """Return value is always a 3-tuple of strings."""
    result = infer_bids_channel_info("SomeUnknownColumn")
    assert isinstance(result, tuple)
    assert len(result) == 3
    for val in result:
        assert isinstance(val, str)


def test_infer_bids_channel_info_unknown_column_returns_misc():
    """Completely unknown column defaults to MISC type."""
    ctype, _, _ = infer_bids_channel_info("CompletelyUnknownColumn_ABC")
    assert ctype == "MISC"


def test_infer_bids_channel_info_latency_units_seconds():
    """Latency channels have units 's'."""
    _, _, units = infer_bids_channel_info("latency")
    assert units == "s"


def test_infer_bids_channel_info_pos_units_meters():
    """Position channels (_px) have units 'm'."""
    _, _, units = infer_bids_channel_info("Node_Head_px")
    assert units == "m"


# ===========================================================================
# count_channel_types
# ===========================================================================


class TestCountChannelTypes:
    def test_returns_dict(self):
        """count_channel_types returns a dict."""
        result = count_channel_types(HEAD_COLS[1:])  # exclude timestamp
        assert isinstance(result, dict)

    def test_pos_count_correct(self):
        """3 position columns → POSChannelCount == 3."""
        pos_cols = ["Node_Head_px", "Node_Head_py", "Node_Head_pz"]
        result = count_channel_types(pos_cols)
        assert result["POSChannelCount"] == 3

    def test_ornt_count_correct(self):
        """4 quaternion columns → ORNTChannelCount == 4."""
        ornt_cols = ["Node_Head_qx", "Node_Head_qy", "Node_Head_qz", "Node_Head_qw"]
        result = count_channel_types(ornt_cols)
        assert result["ORNTChannelCount"] == 4

    def test_timestamp_excluded(self):
        """'timestamp' and 'timeSinceStartup' are not counted."""
        result = count_channel_types(["timestamp", "timeSinceStartup", "Node_Head_px"])
        total = sum(result.values())
        assert total == 1  # only Node_Head_px

    def test_all_expected_keys_present(self):
        """Result dict has keys for all BIDS channel types."""
        from resxr.core.constants import BIDS_CHANNEL_TYPE_COUNTS

        result = count_channel_types([])
        for key in BIDS_CHANNEL_TYPE_COUNTS.values():
            assert key in result


# ===========================================================================
# count_tracked_points
# ===========================================================================


class TestCountTrackedPoints:
    def test_head_xyz_counts_as_one_point(self):
        """Node_Head_px/py/pz + qw all strip to 'Node_Head' → 1 tracked point."""
        cols = ["Node_Head_px", "Node_Head_py", "Node_Head_pz", "Node_Head_qw"]
        assert count_tracked_points(cols) == 1

    def test_left_and_right_hand_are_two_points(self):
        """LeftHand_Root and RightHand_Root strip to different base names → 2 points."""
        cols = [
            "LeftHand_Root_px",
            "LeftHand_Root_py",
            "RightHand_Root_px",
            "RightHand_Root_py",
        ]
        result = count_tracked_points(cols)
        assert result == 2

    def test_empty_list_returns_zero(self):
        """Empty column list → 0 tracked points."""
        assert count_tracked_points([]) == 0

    def test_time_columns_excluded(self):
        """timestamp, timeSinceStartup, latency, latency_global are not points."""
        cols = ["timestamp", "timeSinceStartup", "latency", "latency_global"]
        assert count_tracked_points(cols) == 0
