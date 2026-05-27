"""Tests for HandsTrackingLossCheck (validation/checks/hands_tracking_loss.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from resxr.core.config import ValidationConfig
from resxr.core.constants import TrackingSystem
from resxr.core.session import TrackingStream
from resxr.validation.checks.hands_tracking_loss import HandsTrackingLossCheck
from tests.conftest import make_timestamps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**settings) -> ValidationConfig:
    return ValidationConfig(enabled_checks=["hands_tracking_loss"], settings=settings)


def _make_hands_stream(
    n: int = 100,
    left_tracked: np.ndarray | None = None,
    right_tracked: np.ndarray | None = None,
) -> TrackingStream:
    """Build a HANDS stream with configurable validity columns.

    Column value 1 = tracking valid, 0 = tracking lost.
    """
    ts = make_timestamps(n, 90.0, 1.0)
    if left_tracked is None:
        left_tracked = np.ones(n, dtype=int)
    if right_tracked is None:
        right_tracked = np.ones(n, dtype=int)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "LeftHand_Root_px": np.zeros(n),
            "RightHand_Root_px": np.zeros(n),
            "LeftHand_Status_HandTracked": left_tracked,
            "RightHand_Status_HandTracked": right_tracked,
        }
    )
    return TrackingStream(
        system=TrackingSystem.HANDS,
        data=df,
        sampling_frequency=90.0,
    )


# ===========================================================================
# Metadata
# ===========================================================================


class TestHandsTrackingLossMetadata:
    def test_name(self):
        check = HandsTrackingLossCheck()
        assert check.name == "hands_tracking_loss"

    def test_description_is_string(self):
        check = HandsTrackingLossCheck()
        assert isinstance(check.description, str)
        assert len(check.description) > 0

    def test_required_streams_is_hands(self):
        check = HandsTrackingLossCheck()
        assert check.required_streams == [TrackingSystem.HANDS]


# ===========================================================================
# No tracking loss
# ===========================================================================


class TestHandsTrackingLossNoLoss:
    def test_all_valid_no_flags(self, full_session):
        """All validity columns = 1 → no flags."""
        stream = _make_hands_stream(n=100)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert flags == []

    def test_empty_stream_no_flags(self, full_session):
        """Empty DataFrame returns empty list."""
        df = pd.DataFrame(columns=["timestamp", "LeftHand_Status_HandTracked"])
        stream = TrackingStream(
            system=TrackingSystem.HANDS,
            data=df,
            sampling_frequency=90.0,
        )
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert flags == []


# ===========================================================================
# Tracking loss detection
# ===========================================================================


class TestHandsTrackingLossDetection:
    def test_left_hand_loss_produces_flags(self, full_session):
        """Zeroed left-hand validity → at least one flag for left_hand."""
        left = np.ones(100, dtype=int)
        left[20:30] = 0  # rows 20–29 lost
        stream = _make_hands_stream(left_tracked=left)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        left_flags = [f for f in flags if f.group_name == "left_hand"]
        assert len(left_flags) >= 1

    def test_right_hand_loss_produces_flags(self, full_session):
        """Zeroed right-hand validity → at least one flag for right_hand."""
        right = np.ones(100, dtype=int)
        right[40:55] = 0
        stream = _make_hands_stream(right_tracked=right)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        right_flags = [f for f in flags if f.group_name == "right_hand"]
        assert len(right_flags) >= 1

    def test_both_hands_loss_produces_flags_for_both(self, full_session):
        """Both hands lost → flags for both left_hand and right_hand groups."""
        left = np.ones(100, dtype=int)
        left[10:20] = 0
        right = np.ones(100, dtype=int)
        right[50:60] = 0
        stream = _make_hands_stream(left_tracked=left, right_tracked=right)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        group_names = {f.group_name for f in flags}
        assert "left_hand" in group_names
        assert "right_hand" in group_names

    def test_flag_mask_is_true(self, full_session):
        """Tracking-loss flags have mask=True (data should be NaN-masked)."""
        left = np.ones(100, dtype=int)
        left[10:20] = 0
        stream = _make_hands_stream(left_tracked=left)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert all(f.mask is True for f in flags)

    def test_flag_severity_is_warning(self, full_session):
        """Tracking-loss flags have severity='warning'."""
        left = np.ones(100, dtype=int)
        left[5:15] = 0
        stream = _make_hands_stream(left_tracked=left)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert all(f.severity == "warning" for f in flags)

    def test_flag_check_name(self, full_session):
        """Flag check_name == 'hands_tracking_loss'."""
        left = np.ones(100, dtype=int)
        left[5:15] = 0
        stream = _make_hands_stream(left_tracked=left)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert all(f.check_name == "hands_tracking_loss" for f in flags)

    def test_flag_target_columns_set(self, full_session):
        """Each flag has the tracked column in its target_columns."""
        left = np.ones(100, dtype=int)
        left[10:20] = 0
        stream = _make_hands_stream(left_tracked=left)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        for flag in flags:
            assert len(flag.target_columns) > 0

    def test_flag_time_range_covers_loss_period(self, full_session):
        """Flag start_time / end_time bracket the lost rows."""
        ts = make_timestamps(100, 90.0, 1.0)
        left = np.ones(100, dtype=int)
        left[20:30] = 0
        stream = _make_hands_stream(left_tracked=left)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        left_flags = [f for f in flags if f.group_name == "left_hand"]
        assert len(left_flags) >= 1
        flag = left_flags[0]
        # Flag should start at or before ts[20] and end at or after ts[29]
        assert flag.start_time <= ts[20] + 1e-9
        assert flag.end_time >= ts[29] - 1e-9

    def test_nan_validity_treated_as_lost(self, full_session):
        """NaN in a validity column is also considered tracking lost."""
        ts = make_timestamps(100, 90.0, 1.0)
        left = np.ones(100, dtype=float)
        left[30:35] = np.nan
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "timeSinceStartup": ts,
                "LeftHand_Root_px": np.zeros(100),
                "RightHand_Root_px": np.zeros(100),
                "LeftHand_Status_HandTracked": left,
                "RightHand_Status_HandTracked": np.ones(100, dtype=float),
            }
        )
        stream = TrackingStream(
            system=TrackingSystem.HANDS,
            data=df,
            sampling_frequency=90.0,
        )
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        left_flags = [f for f in flags if f.group_name == "left_hand"]
        assert len(left_flags) >= 1

    def test_all_invalid_produces_flags(self, full_session):
        """Entire stream lost → flags cover the whole recording."""
        left = np.zeros(100, dtype=int)
        right = np.zeros(100, dtype=int)
        stream = _make_hands_stream(left_tracked=left, right_tracked=right)
        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert len(flags) >= 2  # at least one for each hand


# ===========================================================================
# Configurable tracking_flags
# ===========================================================================


class TestHandsTrackingLossConfigurable:
    def test_custom_tracking_flags_column(self, full_session):
        """Custom tracking_flags in config are used instead of defaults."""
        n = 100
        ts = make_timestamps(n, 90.0, 1.0)
        validity = np.ones(n, dtype=int)
        validity[10:20] = 0
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "timeSinceStartup": ts,
                "MyCustomValidity": validity,
            }
        )
        stream = TrackingStream(
            system=TrackingSystem.HANDS,
            data=df,
            sampling_frequency=90.0,
        )
        config = _make_config(tracking_flags={"custom_group": ["MyCustomValidity"]})
        check = HandsTrackingLossCheck()
        flags = check(stream, full_session, config)
        custom_flags = [f for f in flags if f.group_name == "custom_group"]
        assert len(custom_flags) >= 1

    def test_missing_validity_columns_produces_no_flags(self, full_session):
        """If configured columns are absent from the stream, no flags are produced."""
        n = 50
        ts = make_timestamps(n, 90.0, 1.0)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "timeSinceStartup": ts,
                "LeftHand_Root_px": np.zeros(n),
                # Missing LeftHand_Status_HandTracked and RightHand_Status_HandTracked
            }
        )
        stream = TrackingStream(
            system=TrackingSystem.HANDS,
            data=df,
            sampling_frequency=90.0,
        )
        check = HandsTrackingLossCheck()
        config = _make_config()  # default columns not present
        flags = check(stream, full_session, config)
        assert flags == []


# ===========================================================================
# Clock dropout (core bug regression)
# ===========================================================================


class TestHandsTrackingLossClockDropout:
    def test_per_system_clock_dropout_does_not_crash(self, full_session):
        """Per-system timestamp drops to 0 mid-recording; timeSinceStartup keeps ticking.
        This is the exact hardware dropout scenario that previously caused
        start_time > end_time in QualityFlag.__post_init__."""
        n = 100
        ts_per_system = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per_system[30:35] = 0.0  # hardware dropout: per-system clock zeroes out

        left = np.ones(n, dtype=int)
        left[30:35] = 0  # tracking lost during dropout rows

        df = pd.DataFrame(
            {
                "timestamp": ts_per_system,
                "timeSinceStartup": ts_global,
                "LeftHand_Status_HandTracked": left,
                "RightHand_Status_HandTracked": np.ones(n, dtype=int),
            }
        )
        stream = TrackingStream(system=TrackingSystem.HANDS, data=df, sampling_frequency=90.0)

        check = HandsTrackingLossCheck()
        config = _make_config()
        flags = check(stream, full_session, config)  # must not raise

        left_flags = [f for f in flags if f.group_name == "left_hand"]
        assert len(left_flags) >= 1
        for f in left_flags:
            assert f.start_time > 0, (
                "Flag boundaries must use timeSinceStartup, not zeroed timestamp"
            )
            assert f.start_time <= f.end_time

    def test_missing_timeSinceStartup_returns_empty_and_logs_error(self, full_session, caplog):
        """If timeSinceStartup is absent the check must return [] and log an error."""
        import logging

        n = 50
        ts = make_timestamps(n, 90.0, 1.0)
        left = np.zeros(n, dtype=int)  # all tracking lost
        df = pd.DataFrame(
            {
                "timestamp": ts,
                # NO timeSinceStartup column
                "LeftHand_Status_HandTracked": left,
                "RightHand_Status_HandTracked": np.ones(n, dtype=int),
            }
        )
        stream = TrackingStream(system=TrackingSystem.HANDS, data=df, sampling_frequency=90.0)

        check = HandsTrackingLossCheck()
        config = _make_config()
        with caplog.at_level(logging.ERROR):
            flags = check(stream, full_session, config)

        assert flags == []
        assert "timeSinceStartup" in caplog.text
