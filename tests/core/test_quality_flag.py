"""Tests for QualityFlag dataclass (core/session.py)."""

from __future__ import annotations

import numpy as np
import pytest

from resxr.core.constants import TrackingSystem
from resxr.core.session import QualityFlag
from tests.conftest import make_timestamps

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_quality_flag_valid_construction():
    """All fields are stored as-is when start_time < end_time."""
    flag = QualityFlag(
        check_name="my_check",
        system=TrackingSystem.HEAD,
        start_time=1.0,
        end_time=3.5,
        severity="warning",
        message="Test message",
        mask=False,
        group_name="head_group",
        target_columns=["Node_Head_px"],
    )
    assert flag.check_name == "my_check"
    assert flag.system == TrackingSystem.HEAD
    assert flag.start_time == pytest.approx(1.0)
    assert flag.end_time == pytest.approx(3.5)
    assert flag.severity == "warning"
    assert flag.message == "Test message"
    assert flag.mask is False
    assert flag.group_name == "head_group"
    assert flag.target_columns == ["Node_Head_px"]


def test_quality_flag_start_equals_end_is_valid():
    """start_time == end_time is a valid zero-duration flag."""
    flag = QualityFlag(
        check_name="c",
        system=TrackingSystem.HANDS,
        start_time=2.0,
        end_time=2.0,
        severity="warning",
        message="m",
    )
    assert flag.start_time == flag.end_time


def test_quality_flag_start_after_end_raises():
    """start_time > end_time must raise ValueError."""
    with pytest.raises(ValueError, match="start_time.*end_time"):
        QualityFlag(
            check_name="c",
            system=TrackingSystem.HEAD,
            start_time=5.0,
            end_time=1.0,
            severity="warning",
            message="m",
        )


def test_quality_flag_target_columns_default_empty():
    """target_columns defaults to an empty list, not None."""
    flag = QualityFlag(
        check_name="c",
        system=TrackingSystem.HEAD,
        start_time=0.0,
        end_time=1.0,
        severity="warning",
        message="m",
    )
    assert flag.target_columns == []
    assert flag.target_columns is not None


def test_quality_flag_mask_default_true():
    """mask defaults to True when not specified."""
    flag = QualityFlag(
        check_name="c",
        system=TrackingSystem.HEAD,
        start_time=0.0,
        end_time=1.0,
        severity="warning",
        message="m",
    )
    assert flag.mask is True


def test_quality_flag_mask_false_stored():
    """mask=False is stored correctly."""
    flag = QualityFlag(
        check_name="c",
        system=TrackingSystem.HEAD,
        start_time=0.0,
        end_time=1.0,
        severity="warning",
        message="m",
        mask=False,
    )
    assert flag.mask is False


# ---------------------------------------------------------------------------
# duration property
# ---------------------------------------------------------------------------


def test_quality_flag_duration_positive():
    """duration returns end_time - start_time."""
    flag = QualityFlag(
        check_name="c",
        system=TrackingSystem.HEAD,
        start_time=1.0,
        end_time=3.5,
        severity="warning",
        message="m",
    )
    assert flag.duration == pytest.approx(2.5)


def test_quality_flag_duration_zero():
    """duration is 0.0 when start_time == end_time."""
    flag = QualityFlag(
        check_name="c",
        system=TrackingSystem.HEAD,
        start_time=2.0,
        end_time=2.0,
        severity="warning",
        message="m",
    )
    assert flag.duration == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# from_mask classmethod
# ---------------------------------------------------------------------------


def test_from_mask_single_contiguous_segment():
    """One contiguous True region → exactly one flag with correct times."""
    ts = make_timestamps(100, 90.0, 1.0)
    mask = np.zeros(100, dtype=bool)
    mask[10:21] = True  # rows 10–20 inclusive

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="test",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )

    assert len(flags) == 1
    assert flags[0].start_time == pytest.approx(ts[10])
    assert flags[0].end_time == pytest.approx(ts[20])


def test_from_mask_two_disjoint_segments():
    """Two separate True regions → two flags."""
    ts = make_timestamps(100, 90.0, 1.0)
    mask = np.zeros(100, dtype=bool)
    mask[10:15] = True
    mask[50:55] = True

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="test",
        system=TrackingSystem.HANDS,
        severity="error",
        message="m",
    )

    assert len(flags) == 2
    assert flags[0].start_time == pytest.approx(ts[10])
    assert flags[0].end_time == pytest.approx(ts[14])
    assert flags[1].start_time == pytest.approx(ts[50])
    assert flags[1].end_time == pytest.approx(ts[54])


def test_from_mask_all_false_returns_empty():
    """All-False mask → empty list, no flags."""
    ts = make_timestamps(50, 90.0, 1.0)
    mask = np.zeros(50, dtype=bool)

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="test",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert flags == []


def test_from_mask_all_true_returns_one_flag():
    """All-True mask → exactly one flag spanning the full range."""
    ts = make_timestamps(50, 90.0, 1.0)
    mask = np.ones(50, dtype=bool)

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="test",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )

    assert len(flags) == 1
    assert flags[0].start_time == pytest.approx(ts[0])
    assert flags[0].end_time == pytest.approx(ts[-1])


def test_from_mask_single_sample_true():
    """Only one index True → flag where start_time == end_time."""
    ts = make_timestamps(50, 90.0, 1.0)
    mask = np.zeros(50, dtype=bool)
    mask[25] = True

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="test",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )

    assert len(flags) == 1
    assert flags[0].start_time == pytest.approx(ts[25])
    assert flags[0].end_time == pytest.approx(ts[25])


def test_from_mask_kwargs_propagated_to_all_flags():
    """Extra keyword args (severity, should_mask, group_name, target_columns) are set on every flag."""
    ts = make_timestamps(100, 90.0, 1.0)
    mask = np.zeros(100, dtype=bool)
    mask[5:10] = True
    mask[60:70] = True

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="my_check",
        system=TrackingSystem.HANDS,
        severity="error",
        message="tracking lost",
        should_mask=True,
        group_name="left_hand",
        target_columns=["LeftHand_Root_px"],
    )

    assert len(flags) == 2
    for f in flags:
        assert f.check_name == "my_check"
        assert f.system == TrackingSystem.HANDS
        assert f.severity == "error"
        assert f.mask is True
        assert f.group_name == "left_hand"
        assert f.target_columns == ["LeftHand_Root_px"]


def test_from_mask_segment_at_start():
    """Segment beginning at index 0 is found correctly."""
    ts = make_timestamps(50, 90.0, 1.0)
    mask = np.zeros(50, dtype=bool)
    mask[0:5] = True

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )

    assert len(flags) == 1
    assert flags[0].start_time == pytest.approx(ts[0])
    assert flags[0].end_time == pytest.approx(ts[4])


def test_from_mask_segment_at_end():
    """Segment ending at the last index is found correctly."""
    ts = make_timestamps(50, 90.0, 1.0)
    mask = np.zeros(50, dtype=bool)
    mask[45:] = True

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )

    assert len(flags) == 1
    assert flags[0].end_time == pytest.approx(ts[49])


def test_from_mask_length_mismatch_raises():
    """Mismatched timestamps / boolean_mask lengths raise ValueError."""
    ts = make_timestamps(50, 90.0, 1.0)
    mask = np.ones(40, dtype=bool)

    with pytest.raises(ValueError):
        QualityFlag.from_mask(
            timestamps=ts,
            boolean_mask=mask,
            check_name="t",
            system=TrackingSystem.HEAD,
            severity="warning",
            message="m",
        )


# ---------------------------------------------------------------------------
# _find_contiguous_segments static method
# ---------------------------------------------------------------------------


def test_find_contiguous_segments_basic():
    """Known mask returns expected (start_idx, end_idx) index pairs."""
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([False, True, True, False, True, False])
    segments = QualityFlag._find_contiguous_segments(ts, mask)
    assert len(segments) == 2
    assert segments[0] == pytest.approx((1.0, 2.0))
    assert segments[1] == pytest.approx((4.0, 4.0))


def test_find_contiguous_segments_boundary_start():
    """Segment starting at index 0 is included."""
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([True, True, False, False])
    segs = QualityFlag._find_contiguous_segments(ts, mask)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((0.0, 1.0))


def test_find_contiguous_segments_boundary_end():
    """Segment ending at the last index is included."""
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    mask = np.array([False, False, True, True])
    segs = QualityFlag._find_contiguous_segments(ts, mask)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((2.0, 3.0))


def test_find_contiguous_segments_all_false():
    """All-False mask returns empty list."""
    ts = np.array([0.0, 1.0, 2.0])
    mask = np.array([False, False, False])
    assert QualityFlag._find_contiguous_segments(ts, mask) == []


def test_find_contiguous_segments_single_element_true():
    """Single True element gives a single-point segment."""
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = np.array([False, False, True, False, False])
    segs = QualityFlag._find_contiguous_segments(ts, mask)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((2.0, 2.0))


# ---------------------------------------------------------------------------
# from_mask: trailing / leading zero filtering
# ---------------------------------------------------------------------------


def test_from_mask_trailing_zero_segment_dropped():
    """Flag on trailing-zero rows is dropped entirely."""
    ts = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    mask = np.array([False, False, False, False, False, True, True])

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert flags == []


def test_from_mask_leading_zero_segment_dropped():
    """Flag on leading-zero rows is dropped entirely."""
    ts = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    mask = np.array([True, True, False, False, False])

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert flags == []


def test_from_mask_valid_segment_preserved_with_trailing_zeros():
    """A flag in the valid region is kept even when trailing zeros exist."""
    ts = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    mask = np.array([False, True, True, False, False, False])

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert len(flags) == 1
    assert flags[0].start_time == pytest.approx(1.0)
    assert flags[0].end_time == pytest.approx(2.0)


def test_from_mask_segment_spanning_valid_and_trailing_zeros_clamped():
    """A flag that starts in valid data and extends into trailing zeros is trimmed."""
    ts = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    mask = np.array([False, False, True, True, True])

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert len(flags) == 1
    # Only the valid row (index 2, ts=3.0) should remain
    assert flags[0].start_time == pytest.approx(3.0)
    assert flags[0].end_time == pytest.approx(3.0)


def test_from_mask_segment_spanning_leading_zeros_and_valid_clamped():
    """A flag that starts in leading zeros and extends into valid data is clamped."""
    ts = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    mask = np.array([True, True, True, False, False])

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert len(flags) == 1
    # start_time should be clamped to first valid timestamp (1.0)
    assert flags[0].start_time == pytest.approx(1.0)
    assert flags[0].end_time == pytest.approx(1.0)


def test_from_mask_all_zeros_returns_empty():
    """All-zero timestamps with True mask returns no flags."""
    ts = np.zeros(5)
    mask = np.ones(5, dtype=bool)

    flags = QualityFlag.from_mask(
        timestamps=ts,
        boolean_mask=mask,
        check_name="t",
        system=TrackingSystem.HEAD,
        severity="warning",
        message="m",
    )
    assert flags == []
