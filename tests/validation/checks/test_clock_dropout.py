"""Tests for ClockDropoutCheck (validation/checks/clock_dropout.py)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from resxr.core.config import ValidationConfig
from resxr.core.constants import TrackingSystem
from resxr.core.session import TrackingStream
from resxr.validation.checks.clock_dropout import ClockDropoutCheck
from tests.conftest import make_timestamps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> ValidationConfig:
    return ValidationConfig(enabled_checks=["clock_dropout"])


def _make_stream(
    n: int = 100,
    ts_per_system: np.ndarray | None = None,
    ts_global: np.ndarray | None = None,
    system: TrackingSystem = TrackingSystem.HEAD,
    include_global: bool = True,
) -> TrackingStream:
    """Build a minimal stream with configurable per-system and global clocks."""
    if ts_per_system is None:
        ts_per_system = make_timestamps(n, 90.0, 1.0)
    if ts_global is None:
        ts_global = make_timestamps(n, 90.0, 1.0)

    data: dict = {
        "timestamp": ts_per_system,
        "data_col_a": np.zeros(n),
        "data_col_b": np.ones(n),
    }
    if include_global:
        data["timeSinceStartup"] = ts_global

    return TrackingStream(
        system=system,
        data=pd.DataFrame(data),
        sampling_frequency=90.0,
    )


# ===========================================================================
# Metadata
# ===========================================================================


class TestClockDropoutMetadata:
    def test_name(self):
        assert ClockDropoutCheck().name == "clock_dropout"

    def test_description_is_nonempty_string(self):
        desc = ClockDropoutCheck().description
        assert isinstance(desc, str) and len(desc) > 0

    def test_required_streams_is_none(self):
        """Check runs on every stream — required_streams must be None."""
        assert ClockDropoutCheck().required_streams is None


# ===========================================================================
# No dropout (clean streams)
# ===========================================================================


class TestClockDropoutNoDropout:
    def test_all_valid_timestamps_no_flags(self, full_session):
        """Monotonically increasing timestamps → no flags."""
        stream = _make_stream()
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert flags == []

    def test_empty_dataframe_no_flags(self, full_session):
        """Empty DataFrame → no flags (no crash)."""
        df = pd.DataFrame(columns=["timestamp", "timeSinceStartup", "data_col"])
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=df,
            sampling_frequency=90.0,
        )
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert flags == []


# ===========================================================================
# Dropout detection
# ===========================================================================


class TestClockDropoutDetection:
    def test_single_mid_block_produces_flag(self, full_session):
        """Zeros at rows 30–34 → at least one flag."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per[30:35] = 0.0

        stream = _make_stream(n=n, ts_per_system=ts_per, ts_global=ts_global)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert len(flags) >= 1

    def test_two_mid_blocks_produce_two_flags(self, full_session):
        """Two separate dropout blocks → two flags."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per[10:13] = 0.0
        ts_per[50:55] = 0.0

        stream = _make_stream(n=n, ts_per_system=ts_per, ts_global=ts_global)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert len(flags) == 2

    def test_flag_check_name(self, full_session):
        """All flags have check_name == 'clock_dropout'."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[20:25] = 0.0
        stream = _make_stream(n=n, ts_per_system=ts_per)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert all(f.check_name == "clock_dropout" for f in flags)

    def test_flag_mask_is_true(self, full_session):
        """Dropout flags have mask=True (rows should be NaN-masked)."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[20:25] = 0.0
        stream = _make_stream(n=n, ts_per_system=ts_per)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert all(f.mask is True for f in flags)

    def test_flag_target_columns_empty(self, full_session):
        """target_columns==[] means mask all data columns."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[20:25] = 0.0
        stream = _make_stream(n=n, ts_per_system=ts_per)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert all(f.target_columns == [] for f in flags)

    def test_flag_severity_is_warning(self, full_session):
        """Dropout flags have severity='warning'."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[20:25] = 0.0
        stream = _make_stream(n=n, ts_per_system=ts_per)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert all(f.severity == "warning" for f in flags)

    def test_flag_start_le_end(self, full_session):
        """start_time <= end_time for every flag."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[30:35] = 0.0
        stream = _make_stream(n=n, ts_per_system=ts_per)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        for f in flags:
            assert f.start_time <= f.end_time

    def test_flag_boundaries_from_timeSinceStartup(self, full_session):
        """Flag start/end come from timeSinceStartup, not the zeroed timestamp."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per[30:35] = 0.0  # per-system clock drops; ts_global[30] ≈ 1.333

        stream = _make_stream(n=n, ts_per_system=ts_per, ts_global=ts_global)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())

        assert len(flags) >= 1
        flag = flags[0]
        # Boundaries must be > 0 (from the global clock), not 0.0 (from the zeroed timestamp)
        assert flag.start_time > 0, (
            "start_time must use timeSinceStartup, not the zeroed per-system timestamp"
        )
        assert flag.end_time > 0
        # Verify the exact timeSinceStartup values at the dropout indices
        assert flag.start_time == pytest.approx(ts_global[30])
        assert flag.end_time == pytest.approx(ts_global[34])

    def test_nan_mid_block_detected(self, full_session):
        """NaN in timestamp column is also treated as a dropout."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[40:43] = np.nan
        stream = _make_stream(n=n, ts_per_system=ts_per)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert len(flags) >= 1


# ===========================================================================
# Leading / trailing zeros excluded
# ===========================================================================


class TestClockDropoutLeadingTrailingExcluded:
    def test_leading_zeros_only_no_flags(self, full_session):
        """Zeros only at the start (device spin-up) → no flags."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per[:5] = 0.0  # leading zeros

        stream = _make_stream(n=n, ts_per_system=ts_per, ts_global=ts_global)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert flags == []

    def test_trailing_zeros_only_no_flags(self, full_session):
        """Zeros only at the end (device spin-down) → no flags."""
        n = 100
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per[-5:] = 0.0  # trailing zeros

        stream = _make_stream(n=n, ts_per_system=ts_per, ts_global=ts_global)
        flags = ClockDropoutCheck()(stream, full_session, _make_config())
        assert flags == []


# ===========================================================================
# Missing timeSinceStartup
# ===========================================================================


class TestClockDropoutMissingGlobalClock:
    def test_missing_timeSinceStartup_returns_empty(self, full_session, caplog):
        """No timeSinceStartup column → returns [] and logs an ERROR."""
        n = 50
        ts_per = make_timestamps(n, 90.0, 1.0)
        ts_per[10:15] = 0.0  # real dropout rows

        stream = _make_stream(n=n, ts_per_system=ts_per, include_global=False)

        with caplog.at_level(logging.ERROR):
            flags = ClockDropoutCheck()(stream, full_session, _make_config())

        assert flags == []
        assert "timeSinceStartup" in caplog.text
