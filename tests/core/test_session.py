"""Tests for TrackingStream and Session dataclasses (core/session.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.core.constants import TrackingSystem
from resxr.core.session import (
    ColumnInfoEntry,
    CustomTableSchema,
    QualityFlag,
    Session,
    TrackingStream,
)
from tests.conftest import _head_df, make_timestamps

# ===========================================================================
# TrackingStream
# ===========================================================================


class TestTrackingStreamConstruction:
    def test_requires_nonzero_sampling_frequency(self):
        """TrackingStream raises ValueError if sampling_frequency is 0."""
        with pytest.raises(ValueError, match="sampling_frequency"):
            TrackingStream(
                system=TrackingSystem.HEAD,
                data=_head_df(),
                sampling_frequency=0.0,
            )

    def test_channel_count_set_on_construction(self, head_stream):
        """channel_count is computed from non-time columns during __post_init__."""
        # _head_df has: timestamp (time col), Node_Head_px/py/pz/qx/qy/qz/qw → 7 channels
        assert head_stream.channel_count == 7

    def test_effective_rate_computed_on_construction(self, head_stream):
        """sampling_frequency_effective is set automatically from timestamps."""
        assert head_stream.sampling_frequency_effective > 0.0


class TestTrackingStreamProperties:
    def test_row_count(self, head_stream):
        """row_count returns number of rows in data."""
        assert head_stream.row_count == 200

    def test_duration_seconds(self, head_stream):
        """duration_seconds = max(timestamp) - first non-zero timestamp."""
        # 200 rows at 90 Hz starting at 1.0: span = 199/90 seconds
        expected = 199 / 90.0
        assert head_stream.duration_seconds == pytest.approx(expected, rel=1e-5)

    def test_start_timestamp_returns_first_nonzero(self, head_stream):
        """_start_timestamp returns the first non-zero timestamp value."""
        assert head_stream._start_timestamp() == pytest.approx(1.0)

    def test_effective_rate_uniform_equals_nominal(self, head_stream):
        """Effective rate equals configured rate for perfectly uniform timestamps."""
        assert head_stream.sampling_frequency_effective == pytest.approx(90.0, rel=1e-5)

    def test_effective_rate_different_span(self, session_metadata):
        """A stream spanning 3 s instead of ~2.2 s gives a different effective rate."""
        ts = np.linspace(1.0, 4.0, 200)  # 3 s span → ~66.3 Hz effective
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "Node_Head_px": np.zeros(200),
            }
        )
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=df,
            sampling_frequency=90.0,
        )
        assert stream.sampling_frequency_effective != pytest.approx(90.0, rel=0.05)
        assert stream.sampling_frequency_effective == pytest.approx(199 / 3.0, rel=1e-4)

    def test_interval_ms_expected(self, head_stream):
        """interval_ms_expected = 1000 / sampling_frequency."""
        assert head_stream.interval_ms_expected == pytest.approx(1000.0 / 90.0, rel=1e-5)

    def test_interval_ms_effective(self, head_stream):
        """interval_ms_effective = 1000 / sampling_frequency_effective."""
        expected = 1000.0 / head_stream.sampling_frequency_effective
        assert head_stream.interval_ms_effective == pytest.approx(expected, rel=1e-5)

    def test_interval_ms_zero_when_no_effective_rate(self):
        """interval_ms_effective returns 0 if effective rate is 0."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=_head_df(),
            sampling_frequency=90.0,
        )
        stream.sampling_frequency_effective = 0.0
        assert stream.interval_ms_effective == 0.0

    def test_warning_count(self, stream_with_flags):
        """warning_count counts only 'warning'-severity flags."""
        assert stream_with_flags.warning_count == 2

    def test_error_count(self, stream_with_flags):
        """error_count counts only 'error'-severity flags."""
        assert stream_with_flags.error_count == 1

    def test_counts_zero_for_no_flags(self, head_stream):
        """Both counts are 0 when the stream has no flags."""
        assert head_stream.warning_count == 0
        assert head_stream.error_count == 0

    def test_get_output_data_no_clean_returns_raw(self, head_stream):
        """When clean_data is None, get_output_data returns the raw data."""
        assert head_stream.clean_data is None
        result = head_stream.get_output_data()
        assert result is head_stream.data

    def test_get_output_data_with_clean_returns_clean(self, head_stream):
        """When clean_data is set, get_output_data returns clean_data."""
        clean = head_stream.data.copy()
        clean["Node_Head_px"] = 0.0
        head_stream.clean_data = clean
        assert head_stream.get_output_data() is clean


class TestTrackingStreamEdgeCases:
    def test_duration_seconds_empty_dataframe(self, session_metadata):
        """duration_seconds returns 0 for an empty DataFrame without raising."""
        # empty df causes __post_init__ to skip rate computation, so no error
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=pd.DataFrame(),
            sampling_frequency=90.0,
        )
        # __post_init__ checks if data is empty before computing rate
        assert stream.duration_seconds == 0.0

    def test_start_timestamp_none_for_empty_data(self):
        """_start_timestamp returns None when data is empty."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=pd.DataFrame(),
            sampling_frequency=90.0,
        )
        assert stream._start_timestamp() is None


# ===========================================================================
# Session
# ===========================================================================


class TestSessionStreamAccess:
    def test_has_stream_present(self, full_session):
        """has_stream returns True for a stream that exists."""
        assert full_session.has_stream(TrackingSystem.HEAD) is True

    def test_has_stream_absent(self, minimal_session):
        """has_stream returns False for a missing stream."""
        assert minimal_session.has_stream(TrackingSystem.BODY) is False

    def test_get_stream_present(self, full_session):
        """get_stream returns the TrackingStream when it exists."""
        stream = full_session.get_stream(TrackingSystem.HEAD)
        assert stream is not None
        assert isinstance(stream, TrackingStream)
        assert stream.system == TrackingSystem.HEAD

    def test_get_stream_absent_returns_none(self, minimal_session):
        """get_stream returns None (not raises) for a missing stream."""
        result = minimal_session.get_stream(TrackingSystem.BODY)
        assert result is None


class TestSessionFlagAggregation:
    def test_all_flags_sorted_by_start_time(self, session_with_flags):
        """all_flags returns flags from all streams sorted ascending by start_time."""
        flags = session_with_flags.all_flags
        assert len(flags) > 0
        times = [f.start_time for f in flags]
        assert times == sorted(times)

    def test_all_flags_empty_when_no_flags(self, full_session):
        """all_flags is an empty list when no stream has flags."""
        flags = full_session.all_flags
        assert flags == []

    def test_all_flags_cross_stream(self, full_session):
        """all_flags aggregates flags from multiple streams."""
        ts = make_timestamps(200, 90.0, 1.0)
        # Add flags to two different streams
        full_session.streams[TrackingSystem.HEAD].quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=float(ts[5]),
                end_time=float(ts[10]),
                severity="warning",
                message="m",
            )
        ]
        full_session.streams[TrackingSystem.HANDS].quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HANDS,
                start_time=float(ts[1]),
                end_time=float(ts[3]),
                severity="warning",
                message="m",
            )
        ]
        flags = full_session.all_flags
        assert len(flags) == 2
        # HANDS flag (earlier time) should come first
        assert flags[0].system == TrackingSystem.HANDS


class TestSessionMetrics:
    def test_total_duration_seconds_is_max_across_streams(self, full_session):
        """total_duration_seconds returns the MAX stream duration (not sum)."""
        # FACE is at 30 Hz (200 rows), HEAD/HANDS/EYES at 90 Hz (200 rows)
        # FACE duration = 199/30 ≈ 6.63 s  >  HEAD duration = 199/90 ≈ 2.21 s
        expected_max = 199 / 30.0  # FACE is the longest
        assert full_session.total_duration_seconds == pytest.approx(expected_max, rel=1e-4)

    def test_total_duration_seconds_empty_session(self, session_metadata):
        """total_duration_seconds returns 0 for a Session with no streams."""
        session = Session(session_id="empty", metadata=session_metadata, streams={})
        assert session.total_duration_seconds == 0.0

    def test_total_warning_count(self, full_session):
        """total_warning_count sums warning_count across all streams."""
        ts = make_timestamps(200, 90.0, 1.0)
        for stream in full_session.streams.values():
            stream.quality_flags = [
                QualityFlag(
                    check_name="c",
                    system=stream.system,
                    start_time=float(ts[5]),
                    end_time=float(ts[10]),
                    severity="warning",
                    message="m",
                    mask=False,
                )
            ]
        assert full_session.total_warning_count == len(full_session.streams)

    def test_total_error_count(self, full_session):
        """total_error_count sums error_count across all streams."""
        ts = make_timestamps(200, 90.0, 1.0)
        for stream in full_session.streams.values():
            stream.quality_flags = [
                QualityFlag(
                    check_name="c",
                    system=stream.system,
                    start_time=float(ts[5]),
                    end_time=float(ts[10]),
                    severity="error",
                    message="m",
                    mask=True,
                )
            ]
        assert full_session.total_error_count == len(full_session.streams)

    def test_masked_time_seconds_no_masking_flags(self, session_with_flags):
        """masked_time_seconds = 0 when all flags have mask=False."""
        assert session_with_flags.masked_time_seconds == pytest.approx(0.0)

    def test_masked_time_seconds_with_masking_flags(self, minimal_session):
        """masked_time_seconds sums durations of mask=True flags (merged overlaps)."""
        make_timestamps(200, 90.0, 1.0)
        minimal_session.streams[TrackingSystem.HEAD].quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=1.0,
                end_time=2.5,
                severity="error",
                message="m",
                mask=True,
            )
        ]
        assert minimal_session.masked_time_seconds == pytest.approx(1.5)

    def test_masked_time_seconds_overlapping_flags_merged(self, minimal_session):
        """Overlapping masking intervals are merged before summing."""
        minimal_session.streams[TrackingSystem.HEAD].quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=1.0,
                end_time=3.0,
                severity="error",
                message="m1",
                mask=True,
            ),
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=2.0,
                end_time=4.0,
                severity="error",
                message="m2",
                mask=True,
            ),
        ]
        # Merged span = [1.0, 4.0] → 3.0 s (not 2.0 + 2.0 = 4.0 s)
        assert minimal_session.masked_time_seconds == pytest.approx(3.0)

    def test_masked_percentage_correct(self, minimal_session):
        """masked_percentage = (masked_time / total_duration) * 100."""
        ts = make_timestamps(200, 90.0, 1.0)
        total = minimal_session.total_duration_seconds  # ≈ 199/90
        mask_duration = total / 2.0

        minimal_session.streams[TrackingSystem.HEAD].quality_flags = [
            QualityFlag(
                check_name="c",
                system=TrackingSystem.HEAD,
                start_time=float(ts[0]),
                end_time=float(ts[0]) + mask_duration,
                severity="error",
                message="m",
                mask=True,
            )
        ]
        assert minimal_session.masked_percentage == pytest.approx(50.0, abs=1.0)

    def test_masked_percentage_zero_when_no_duration(self, session_metadata):
        """masked_percentage returns 0 when total_duration_seconds is 0."""
        session = Session(session_id="empty", metadata=session_metadata, streams={})
        assert session.masked_percentage == 0.0


# ===========================================================================
# Custom-tables data model (ColumnInfoEntry, CustomTableSchema, Session fields)
# ===========================================================================


class TestColumnInfoEntry:
    def test_minimal_only_description_and_format(self):
        """description and format are required; everything else defaults to None."""
        c = ColumnInfoEntry(name="rt", description="Reaction time", format="float")
        assert c.units is None
        assert c.levels is None
        assert c.minimum is None
        assert c.maximum is None

    def test_units_without_levels(self):
        c = ColumnInfoEntry(name="rt", description="Reaction time", format="float", units="s")
        assert c.units == "s"
        assert c.levels is None

    def test_levels_without_units(self):
        c = ColumnInfoEntry(
            name="resp", description="Response", format="str", levels={"L": "left", "R": "right"}
        )
        assert c.levels == {"L": "left", "R": "right"}
        assert c.units is None

    def test_minimum_and_maximum(self):
        c = ColumnInfoEntry(
            name="score", description="Score", format="float", minimum=0.0, maximum=1.0
        )
        assert c.minimum == 0.0
        assert c.maximum == 1.0


class TestCustomTableSchema:
    def test_holds_columns(self):
        schema = CustomTableSchema(
            class_name="ChoiceEvent",
            row_count=3,
            columns=[ColumnInfoEntry(name="rt", description="Reaction time", format="float")],
        )
        assert schema.class_name == "ChoiceEvent"
        assert schema.row_count == 3
        assert len(schema.columns) == 1


class TestSessionCustomTableFields:
    def test_defaults(self):
        """New Session fields default to empty/None and do not require arguments."""
        s = Session(session_id="s1")
        assert s.custom_tables is None
        assert s.custom_tables_data == {}
        assert s.merged_events_data is None
        assert s.session_flags == []
