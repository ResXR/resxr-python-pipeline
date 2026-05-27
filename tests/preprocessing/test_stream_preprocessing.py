"""Tests for stream preprocessing functions (preprocessing/stream_preprocessing.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.core.constants import TrackingSystem
from resxr.core.session import QualityFlag, TrackingStream
from resxr.preprocessing.stream_preprocessing import (
    apply_quality_masking,
    prepare_motion_data,
    preprocess_stream,
)
from tests.conftest import _head_df, make_timestamps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream_with_masking_flag(
    start_row: int = 10,
    end_row: int = 20,
    target_columns: list[str] | None = None,
    check_name: str = "masking_check",
) -> TrackingStream:
    """Create a HEAD stream with one mask=True flag spanning rows [start_row, end_row]."""
    stream = TrackingStream(
        system=TrackingSystem.HEAD,
        data=_head_df(200),
        sampling_frequency=90.0,
    )
    ts = make_timestamps(200, 90.0, 1.0)
    stream.quality_flags = [
        QualityFlag(
            check_name=check_name,
            system=TrackingSystem.HEAD,
            start_time=float(ts[start_row]),
            end_time=float(ts[end_row]),
            severity="error",
            message="masking flag",
            mask=True,
            target_columns=target_columns if target_columns is not None else [],
        )
    ]
    return stream


# ===========================================================================
# apply_quality_masking
# ===========================================================================


class TestApplyQualityMasking:
    def test_no_flags_returns_unchanged_data(self, head_stream):
        """Stream with no flags: result equals original data."""
        result = apply_quality_masking(head_stream)
        pd.testing.assert_frame_equal(result, head_stream.data)

    def test_mask_true_flag_introduces_nans(self):
        """mask=True flag introduces NaN values in the flagged time range."""
        stream = _stream_with_masking_flag(start_row=10, end_row=20)
        result = apply_quality_masking(stream)
        ts = make_timestamps(200, 90.0, 1.0)
        flagged_mask = (result["timestamp"] >= ts[10]) & (result["timestamp"] <= ts[20])
        # All data columns in the flagged range should be NaN
        data_cols = [c for c in result.columns if c not in {"timestamp", "timeSinceStartup"}]
        for col in data_cols:
            assert result.loc[flagged_mask, col].isna().all(), (
                f"Column {col} not NaN in flagged range"
            )

    def test_timestamp_column_never_masked(self):
        """'timestamp' column is never set to NaN, even for mask=True flags."""
        stream = _stream_with_masking_flag(start_row=10, end_row=20)
        result = apply_quality_masking(stream)
        assert result["timestamp"].notna().all()

    def test_timeSinceStartup_never_masked(self):
        """'timeSinceStartup' column is never masked."""
        stream = _stream_with_masking_flag(start_row=10, end_row=20)
        ts_global = make_timestamps(200, 90.0, 1.0)
        stream.data["timeSinceStartup"] = ts_global  # add global time col
        result = apply_quality_masking(stream)
        assert result["timeSinceStartup"].notna().all()

    def test_mask_false_flag_leaves_data_unchanged(self):
        """A flag with mask=False does not introduce any NaN values."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=_head_df(200),
            sampling_frequency=90.0,
        )
        ts = make_timestamps(200, 90.0, 1.0)
        stream.quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=float(ts[10]),
                end_time=float(ts[20]),
                severity="warning",
                message="no masking",
                mask=False,
            )
        ]
        result = apply_quality_masking(stream)
        assert not result.isnull().any().any()

    def test_target_columns_empty_masks_all_data_columns(self):
        """Empty target_columns means ALL data columns are NaN in the flagged range."""
        stream = _stream_with_masking_flag(target_columns=[])  # empty = all
        result = apply_quality_masking(stream)
        ts = make_timestamps(200, 90.0, 1.0)
        flagged = (result["timestamp"] >= ts[10]) & (result["timestamp"] <= ts[20])
        data_cols = [c for c in result.columns if c not in {"timestamp", "timeSinceStartup"}]
        assert len(data_cols) > 0
        for col in data_cols:
            assert result.loc[flagged, col].isna().all()

    def test_target_columns_specified_masks_only_those(self):
        """Non-empty target_columns masks only the listed columns."""
        stream = _stream_with_masking_flag(target_columns=["Node_Head_px"])
        result = apply_quality_masking(stream)
        ts = make_timestamps(200, 90.0, 1.0)
        flagged = (result["timestamp"] >= ts[10]) & (result["timestamp"] <= ts[20])
        # The targeted column should be NaN
        assert result.loc[flagged, "Node_Head_px"].isna().all()
        # Other data columns should NOT be NaN
        for col in ("Node_Head_py", "Node_Head_pz", "Node_Head_qw"):
            assert result.loc[flagged, col].notna().all()

    def test_rows_outside_flag_range_not_modified(self):
        """Data rows before and after the flagged range are unchanged."""
        stream = _stream_with_masking_flag(start_row=50, end_row=60)
        result = apply_quality_masking(stream)
        ts = make_timestamps(200, 90.0, 1.0)
        unflagged = (result["timestamp"] < ts[50]) | (result["timestamp"] > ts[60])
        for col in ("Node_Head_px", "Node_Head_py", "Node_Head_pz"):
            assert result.loc[unflagged, col].notna().all()

    def test_masking_checks_filter_by_name(self):
        """Only flags from the specified check names are applied."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=_head_df(200),
            sampling_frequency=90.0,
        )
        ts = make_timestamps(200, 90.0, 1.0)
        stream.quality_flags = [
            QualityFlag(
                check_name="allowed_check",
                system=TrackingSystem.HEAD,
                start_time=float(ts[10]),
                end_time=float(ts[20]),
                severity="error",
                message="m",
                mask=True,
            ),
            QualityFlag(
                check_name="other_check",
                system=TrackingSystem.HEAD,
                start_time=float(ts[30]),
                end_time=float(ts[40]),
                severity="error",
                message="m",
                mask=True,
            ),
        ]
        result = apply_quality_masking(stream, masking_checks=["allowed_check"])
        flagged_allowed = (result["timestamp"] >= ts[10]) & (result["timestamp"] <= ts[20])
        flagged_other = (result["timestamp"] >= ts[30]) & (result["timestamp"] <= ts[40])
        # allowed_check range should be NaN
        assert result.loc[flagged_allowed, "Node_Head_px"].isna().all()
        # other_check range should NOT be NaN
        assert result.loc[flagged_other, "Node_Head_px"].notna().all()

    def test_multiple_flags_combined(self):
        """Two non-overlapping masking flags each NaN their respective ranges."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=_head_df(200),
            sampling_frequency=90.0,
        )
        ts = make_timestamps(200, 90.0, 1.0)
        stream.quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=float(ts[5]),
                end_time=float(ts[10]),
                severity="error",
                message="m1",
                mask=True,
            ),
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=float(ts[50]),
                end_time=float(ts[55]),
                severity="error",
                message="m2",
                mask=True,
            ),
        ]
        result = apply_quality_masking(stream)
        mask1 = (result["timestamp"] >= ts[5]) & (result["timestamp"] <= ts[10])
        mask2 = (result["timestamp"] >= ts[50]) & (result["timestamp"] <= ts[55])
        assert result.loc[mask1, "Node_Head_px"].isna().all()
        assert result.loc[mask2, "Node_Head_px"].isna().all()

    def test_empty_stream_returns_empty_copy(self):
        """Stream with empty data returns an empty DataFrame without raising."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=pd.DataFrame(),
            sampling_frequency=90.0,
        )
        result = apply_quality_masking(stream)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_masking_checks_none_applies_all_flags(self):
        """masking_checks=None (the default) applies ALL mask=True flags.

        Contract: `if masking_checks is not None` → filter. None skips the filter.
        """
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=_head_df(200),
            sampling_frequency=90.0,
        )
        ts = make_timestamps(200, 90.0, 1.0)
        stream.quality_flags = [
            QualityFlag(
                check_name="check_alpha",
                system=TrackingSystem.HEAD,
                start_time=float(ts[10]),
                end_time=float(ts[15]),
                severity="error",
                message="m",
                mask=True,
            ),
            QualityFlag(
                check_name="check_beta",
                system=TrackingSystem.HEAD,
                start_time=float(ts[30]),
                end_time=float(ts[35]),
                severity="error",
                message="m",
                mask=True,
            ),
        ]
        result = apply_quality_masking(stream, masking_checks=None)
        flagged_alpha = (result["timestamp"] >= ts[10]) & (result["timestamp"] <= ts[15])
        flagged_beta = (result["timestamp"] >= ts[30]) & (result["timestamp"] <= ts[35])
        # Both ranges should be NaN
        assert result.loc[flagged_alpha, "Node_Head_px"].isna().all()
        assert result.loc[flagged_beta, "Node_Head_px"].isna().all()

    def test_masking_checks_empty_list_applies_no_flags(self):
        """masking_checks=[] (empty list) applies NO masking at all.

        Contract: `masking_checks is not None` → True for [] → filters to empty set.
        This is DIFFERENT from None which means 'apply all'.
        """
        stream = _stream_with_masking_flag(start_row=10, end_row=20)
        result = apply_quality_masking(stream, masking_checks=[])
        # No NaN should appear — the empty list filters out every flag
        data_cols = [c for c in result.columns if c not in {"timestamp", "timeSinceStartup"}]
        for col in data_cols:
            assert result[col].notna().all(), f"Column {col} has NaN despite masking_checks=[]"

    def test_full_stream_masking_flag_nans_all_rows(self):
        """A flag spanning the entire stream NaN-ifies every data row."""
        stream = _stream_with_masking_flag(start_row=0, end_row=199)
        result = apply_quality_masking(stream)
        data_cols = [c for c in result.columns if c not in {"timestamp", "timeSinceStartup"}]
        for col in data_cols:
            assert result[col].isna().all(), (
                f"Column {col} not fully NaN despite full-stream masking flag"
            )

    def test_dropout_rows_masked_when_per_system_clock_is_zero(self):
        """Core bug regression: per-system timestamp drops to 0 during dropout while
        timeSinceStartup keeps ticking. Masking must NaN the dropout rows using
        timeSinceStartup, not the zeroed per-system clock."""
        n = 100
        ts_per_system = make_timestamps(n, 90.0, 1.0)
        ts_global = make_timestamps(n, 90.0, 1.0)
        ts_per_system[30:35] = 0.0  # hardware dropout: per-system clock zeroes out

        data = pd.DataFrame(
            {
                "timestamp": ts_per_system,
                "timeSinceStartup": ts_global,
                "Node_Head_px": np.ones(n),
            }
        )
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=data,
            sampling_frequency=90.0,
        )
        # Flag uses timeSinceStartup boundaries (as fixed checks now produce)
        stream.quality_flags = [
            QualityFlag(
                check_name="test",
                system=TrackingSystem.HEAD,
                start_time=float(ts_global[30]),
                end_time=float(ts_global[34]),
                severity="warning",
                message="dropout",
                mask=True,
            )
        ]

        result = apply_quality_masking(stream)

        # Dropout rows (30-34) must be NaN despite having timestamp == 0.0
        assert result.loc[30:34, "Node_Head_px"].isna().all(), (
            "Dropout rows not masked — masking may be comparing against per-system clock"
        )
        # Rows outside dropout must be untouched
        assert result.loc[0:29, "Node_Head_px"].notna().all()
        assert result.loc[35:, "Node_Head_px"].notna().all()

    def test_missing_timeSinceStartup_returns_unmasked_data_and_logs_error(self, caplog):
        """If timeSinceStartup is absent, masking logs an error and returns data unchanged."""
        import logging

        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=_head_df(50).drop(columns=["timeSinceStartup"]),
            sampling_frequency=90.0,
        )
        ts = make_timestamps(50, 90.0, 1.0)
        stream.quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=float(ts[10]),
                end_time=float(ts[20]),
                severity="warning",
                message="m",
                mask=True,
            )
        ]

        with caplog.at_level(logging.ERROR):
            result = apply_quality_masking(stream)

        assert "timeSinceStartup" in caplog.text
        # Data unchanged (error path returns early)
        assert not result.isnull().any().any()


# ===========================================================================
# preprocess_stream
# ===========================================================================


class TestPreprocessStream:
    def test_sets_clean_data_on_stream(self, head_stream):
        """preprocess_stream sets stream.clean_data to a DataFrame."""
        preprocess_stream(head_stream, apply_masking=False)
        assert head_stream.clean_data is not None
        assert isinstance(head_stream.clean_data, pd.DataFrame)

    def test_masking_applied_when_enabled(self):
        """With apply_masking=True, clean_data has NaN in flagged rows."""
        stream = _stream_with_masking_flag(start_row=10, end_row=20)
        preprocess_stream(stream, apply_masking=True)
        ts = make_timestamps(200, 90.0, 1.0)
        flagged = (stream.clean_data["timestamp"] >= ts[10]) & (
            stream.clean_data["timestamp"] <= ts[20]
        )
        assert stream.clean_data.loc[flagged, "Node_Head_px"].isna().all()

    def test_no_masking_when_disabled(self):
        """With apply_masking=False, clean_data has no NaN in the flagged rows."""
        stream = _stream_with_masking_flag(start_row=10, end_row=20)
        preprocess_stream(stream, apply_masking=False)
        # Stream has a masking flag, but we disabled masking
        assert not stream.clean_data.isnull().any().any()


# ===========================================================================
# prepare_motion_data
# ===========================================================================


class TestPrepareMotionData:
    def test_timestamp_converted_to_latency(self, head_stream):
        """'latency' column exists; 'timestamp' is removed from output."""
        result = prepare_motion_data(head_stream.data)
        assert "latency" in result.columns
        assert "timestamp" not in result.columns

    def test_latency_starts_at_zero(self, head_stream):
        """First non-pre-onset row has latency == 0.0."""
        result = prepare_motion_data(head_stream.data)
        first_valid = result["latency"].dropna().iloc[0]
        assert first_valid == pytest.approx(0.0)

    def test_latency_increases_monotonically(self, head_stream):
        """Valid latency values (non-NaN) are non-decreasing."""
        result = prepare_motion_data(head_stream.data)
        valid = result["latency"].dropna().values
        assert (np.diff(valid) >= -1e-9).all()

    def test_latency_is_seconds_not_ms(self, head_stream):
        """Latency is in seconds: for a ~2s recording, max should be < 10."""
        result = prepare_motion_data(head_stream.data)
        max_latency = result["latency"].dropna().max()
        assert max_latency < 10.0  # not milliseconds

    def test_timeSinceStartup_removed(self, head_stream):
        """'timeSinceStartup' is removed from output."""
        # Add a timeSinceStartup column
        df = head_stream.data.copy()
        df["timeSinceStartup"] = df["timestamp"]
        result = prepare_motion_data(df)
        assert "timeSinceStartup" not in result.columns

    def test_latency_global_added_when_timeSinceStartup_present(self, head_stream):
        """'latency_global' column is added when timeSinceStartup is present."""
        df = head_stream.data.copy()
        df["timeSinceStartup"] = df["timestamp"] + 0.5
        result = prepare_motion_data(df)
        assert "latency_global" in result.columns

    def test_latency_global_absent_without_timeSinceStartup(self, head_stream):
        """'latency_global' is NOT added when timeSinceStartup is absent."""
        df = head_stream.data.drop(columns=["timeSinceStartup"])
        result = prepare_motion_data(df)
        assert "latency_global" not in result.columns

    def test_data_columns_preserved(self, head_stream):
        """Non-time data columns (e.g., Node_Head_px) survive prepare_motion_data."""
        result = prepare_motion_data(head_stream.data)
        assert "Node_Head_px" in result.columns
        assert "Node_Head_py" in result.columns

    def test_pre_onset_rows_set_to_nan(self):
        """Rows where raw timestamp was 0 have NaN latency."""
        ts = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "some_col": np.ones(6),
            }
        )
        result = prepare_motion_data(df)
        # Rows where original timestamp == 0 should have NaN latency
        pre_onset = result.index[ts == 0.0]
        assert result.loc[pre_onset, "latency"].isna().all()

    def test_raw_data_not_modified(self, head_stream):
        """Original DataFrame passed to prepare_motion_data is not modified in-place."""
        original_cols = list(head_stream.data.columns)
        _ = prepare_motion_data(head_stream.data)
        assert list(head_stream.data.columns) == original_cols

    def test_post_offset_rows_set_to_nan(self):
        """Rows where raw timestamp drops back to 0 after valid data have NaN latency."""
        ts = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "some_col": np.ones(7),
            }
        )
        result = prepare_motion_data(df)
        # Pre-onset rows (indices 0, 1) should be NaN
        assert result.loc[0, "latency"] != result.loc[0, "latency"]  # NaN
        assert result.loc[1, "latency"] != result.loc[1, "latency"]  # NaN
        # Valid rows (indices 2, 3, 4) should have finite latency
        assert result.loc[2, "latency"] == pytest.approx(0.0)
        assert result.loc[3, "latency"] == pytest.approx(1.0)
        assert result.loc[4, "latency"] == pytest.approx(2.0)
        # Post-offset rows (indices 5, 6) should be NaN
        assert result.loc[5, "latency"] != result.loc[5, "latency"]  # NaN
        assert result.loc[6, "latency"] != result.loc[6, "latency"]  # NaN

    def test_post_offset_timeSinceStartup_set_to_nan(self):
        """Trailing zeros in timeSinceStartup produce NaN in latency_global."""
        ts = np.array([1.0, 2.0, 3.0, 0.0])
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "timeSinceStartup": ts.copy(),
                "some_col": np.ones(4),
            }
        )
        result = prepare_motion_data(df)
        # Last row (trailing zero) should be NaN for both latency columns
        assert result.loc[3, "latency"] != result.loc[3, "latency"]  # NaN
        assert result.loc[3, "latency_global"] != result.loc[3, "latency_global"]  # NaN
        # Valid rows should be finite
        assert result.loc[0, "latency"] == pytest.approx(0.0)
        assert result.loc[0, "latency_global"] == pytest.approx(0.0)

    def test_no_trailing_zeros_unchanged(self, head_stream):
        """Data without trailing zeros produces identical results to before."""
        result = prepare_motion_data(head_stream.data)
        # No NaN except if there were leading zeros (there aren't in head_stream)
        assert result["latency"].notna().all()
