"""Tests for the SamplingRateCheck (validation/checks/sampling_rate.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.core.config import ValidationConfig
from resxr.core.constants import TrackingSystem
from resxr.core.exceptions import DataLoadError
from resxr.core.session import TrackingStream
from resxr.validation.checks.sampling_rate import SamplingRateCheck

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**settings) -> ValidationConfig:
    return ValidationConfig(
        enabled_checks=["sampling_rate"],
        settings=settings,
    )


def _make_stream(
    n: int = 200,
    rate: float = 90.0,
    start: float = 1.0,
    nominal_rate: float = 90.0,
) -> TrackingStream:
    """Build a HEAD stream with uniformly-spaced timestamps at *rate* Hz."""
    ts = np.arange(n, dtype=float) / rate + start
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "Node_Head_px": np.zeros(n),
        }
    )
    return TrackingStream(
        system=TrackingSystem.HEAD,
        data=df,
        sampling_frequency=nominal_rate,
    )


def _make_irregular_stream(n: int = 100, nominal_rate: float = 90.0) -> TrackingStream:
    """Stream with jittered timestamps that produce high coefficient of variation."""
    rng = np.random.default_rng(7)
    # Large random jitter to push CV well above 0.5
    diffs = rng.exponential(scale=1.0 / nominal_rate, size=n)
    ts = np.cumsum(diffs) + 1.0
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "Node_Head_px": np.zeros(n),
        }
    )
    return TrackingStream(
        system=TrackingSystem.HEAD,
        data=df,
        sampling_frequency=nominal_rate,
    )


# ===========================================================================
# Metadata
# ===========================================================================


class TestSamplingRateCheckMetadata:
    def test_name(self):
        check = SamplingRateCheck()
        assert check.name == "sampling_rate"

    def test_description_is_string(self):
        check = SamplingRateCheck()
        assert isinstance(check.description, str)
        assert len(check.description) > 0

    def test_no_required_streams_attribute(self):
        """SamplingRateCheck has no required_streams (runs on each stream independently)."""
        check = SamplingRateCheck()
        assert getattr(check, "required_streams", None) is None


# ===========================================================================
# Perfect / on-spec data
# ===========================================================================


class TestSamplingRateNoFlags:
    def test_perfect_data_no_flags(self, head_stream, full_session, validation_config):
        """Uniform 90 Hz data with a 90 Hz nominal rate produces no flags."""
        check = SamplingRateCheck()
        flags = check(head_stream, full_session, validation_config)
        assert flags == []

    def test_within_tolerance_no_flags(self, full_session):
        """5% deviation is below the default 10% tolerance — no flags."""
        # 90 Hz nominal, 94.5 Hz actual (5% over)
        stream = _make_stream(n=200, rate=94.5, nominal_rate=90.0)
        check = SamplingRateCheck()
        config = _make_config()  # default tolerance=0.10
        flags = check(stream, full_session, config)
        # Only check for rate-mismatch flag (first flag type)
        rate_flags = [f for f in flags if "mismatch" in f.message.lower()]
        assert rate_flags == []


# ===========================================================================
# Rate-mismatch flags
# ===========================================================================


class TestSamplingRateDeviationFlag:
    def test_flag_for_deviation_above_threshold(self, full_session):
        """25% deviation above the default 10% tolerance → flag."""
        stream = _make_stream(n=200, rate=67.5, nominal_rate=90.0)  # 25% under
        check = SamplingRateCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        rate_flags = [f for f in flags if "mismatch" in f.message.lower()]
        assert len(rate_flags) == 1

    def test_flag_severity_is_warning(self, full_session):
        """The rate-mismatch flag has severity='warning'."""
        stream = _make_stream(n=200, rate=50.0, nominal_rate=90.0)
        check = SamplingRateCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        rate_flags = [f for f in flags if "mismatch" in f.message.lower()]
        assert all(f.severity == "warning" for f in rate_flags)

    def test_flag_mask_is_false(self, full_session):
        """Rate-mismatch flags have mask=False (don't NaN-out the data)."""
        stream = _make_stream(n=200, rate=50.0, nominal_rate=90.0)
        check = SamplingRateCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        rate_flags = [f for f in flags if "mismatch" in f.message.lower()]
        assert all(f.mask is False for f in rate_flags)

    @pytest.mark.parametrize(
        "tolerance,should_flag",
        [
            (0.05, True),  # 15% deviation > 5% tolerance → flag
            (0.20, False),  # 15% deviation < 20% tolerance → no flag
        ],
    )
    def test_configurable_tolerance(self, full_session, tolerance, should_flag):
        """sampling_rate_tolerance config setting is respected."""
        stream = _make_stream(n=200, rate=76.5, nominal_rate=90.0)  # ~15% under
        check = SamplingRateCheck()
        config = _make_config(sampling_rate_tolerance=tolerance)
        flags = check(stream, full_session, config)
        rate_flags = [f for f in flags if "mismatch" in f.message.lower()]
        assert bool(rate_flags) == should_flag


# ===========================================================================
# Irregular sampling flags
# ===========================================================================


class TestSamplingRateIrregularityFlag:
    def test_flag_for_highly_irregular_sampling(self, full_session):
        """Highly jittered timestamps produce a CV-irregularity flag."""
        stream = _make_irregular_stream(n=200, nominal_rate=90.0)
        check = SamplingRateCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        cv_flags = [f for f in flags if "irregular" in f.message.lower()]
        assert len(cv_flags) >= 1

    def test_no_cv_flag_for_regular_sampling(self, head_stream, full_session):
        """Perfectly uniform data has CV ≈ 0, so no irregularity flag."""
        check = SamplingRateCheck()
        config = _make_config()
        flags = check(head_stream, full_session, config)
        cv_flags = [f for f in flags if "irregular" in f.message.lower()]
        assert cv_flags == []


def _template_like_stream(sdk_times: np.ndarray, app_times: np.ndarray) -> TrackingStream:
    """A stream with its own SDK clock (``timestamp``) and the application clock."""
    df = pd.DataFrame(
        {
            "timestamp": sdk_times,
            "timeSinceStartup": app_times,
            "Node_Head_px": np.zeros(len(app_times)),
        }
    )
    return TrackingStream(system=TrackingSystem.HEAD, data=df, sampling_frequency=72.0)


def _bursty_rows(n_frames: int = 300):
    """Rows written in per-frame bursts, as the template's FixedUpdate logging does.

    Frames at 72 Hz; 1 or 2 rows per frame (100 rows/s), 1 ms apart within a frame. Each
    row holds the newest head sample, which the SDK delivers once per frame at 72 Hz.
    """
    frames = 1.0 + np.arange(n_frames) / 72.0
    rows_per_frame = np.where(np.arange(n_frames) % 18 < 7, 2, 1)  # 25 rows per 18 frames
    app = np.concatenate(
        [f + 0.001 * np.arange(k) for f, k in zip(frames, rows_per_frame, strict=True)]
    )
    sdk = np.repeat(500.0 + np.arange(n_frames) / 72.0, rows_per_frame)
    return sdk, app


def _cv(times: np.ndarray) -> float:
    steps = np.diff(np.unique(times))
    return float(np.std(steps) / np.mean(steps))


class TestIrregularityUsesTheStreamClock:
    """D11: the irregularity test measures the intervals between the stream's own samples."""

    @staticmethod
    def _irregular(stream, session) -> list:
        flags = SamplingRateCheck()(stream, session, _make_config())
        return [f for f in flags if f.message.startswith("Highly irregular sampling")]

    def test_bursty_rows_with_regular_sdk_samples_give_no_flag(self, full_session):
        sdk, app = _bursty_rows()
        assert _cv(app) > 0.5  # the application clock alone would have flagged it
        assert self._irregular(_template_like_stream(sdk, app), full_session) == []

    def test_irregular_sdk_samples_give_one_flag_spanning_the_recording(self, full_session):
        rng = np.random.default_rng(7)
        app = 1.0 + np.arange(300) / 100.0  # regular rows
        sdk = 500.0 + np.cumsum(rng.exponential(scale=1.0 / 72.0, size=300))
        app[:3] = 0.0  # rows before onset are outside the flag
        (flag,) = self._irregular(_template_like_stream(sdk, app), full_session)
        assert flag.message == f"Highly irregular sampling: CV={_cv(sdk[3:]):.2f}"
        assert flag.message != "Highly irregular sampling: CV=0.00"
        assert (flag.start_time, flag.end_time) == pytest.approx((app[3], app[-1]))

    def test_zero_and_empty_sdk_values_are_not_samples(self, full_session):
        sdk, app = _bursty_rows()
        sdk[:10] = 0.0  # not tracked yet
        sdk[200:210] = 0.0  # a dropout inside the recording (the clock_dropout check's case)
        sdk[100] = np.nan
        assert self._irregular(_template_like_stream(sdk, app), full_session) == []

    def test_a_stream_without_its_own_clock_is_recorded_as_not_run(self, full_session):
        ts = 1.0 + np.arange(100) / 72.0
        df = pd.DataFrame({"timestamp": ts, "Node_Head_px": np.zeros(100)})
        stream = TrackingStream(system=TrackingSystem.HEAD, data=df, sampling_frequency=72.0)
        assert SamplingRateCheck()(stream, full_session, _make_config()) == []
        (not_run,) = full_session.checks_not_run
        assert (not_run.check, not_run.stream) == ("sampling_rate", "Head")


# ===========================================================================
# Edge cases
# ===========================================================================


class TestSamplingRateEdgeCases:
    def test_empty_dataframe_no_crash(self, full_session):
        """Empty DataFrame returns empty list without raising."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=pd.DataFrame(columns=["timestamp", "Node_Head_px"]),
            sampling_frequency=90.0,
        )
        check = SamplingRateCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert flags == []

    def test_single_row_cannot_create_stream(self, full_session):
        """TrackingStream requires ≥2 rows to compute effective rate.

        This documents that the check is never invoked on a 1-row stream because
        the TrackingStream constructor itself prevents it.
        """
        with pytest.raises(DataLoadError, match="at least 2 rows"):
            TrackingStream(
                system=TrackingSystem.HEAD,
                data=pd.DataFrame({"timestamp": [1.0], "Node_Head_px": [0.0]}),
                sampling_frequency=90.0,
            )

    def test_missing_timestamp_column_cannot_create_stream(self, full_session):
        """TrackingStream requires a 'timestamp' column.

        This documents that the check is never invoked on a stream without
        timestamps because the TrackingStream constructor itself prevents it.
        """
        with pytest.raises(DataLoadError, match="timestamp"):
            TrackingStream(
                system=TrackingSystem.HEAD,
                data=pd.DataFrame({"Node_Head_px": [0.0, 1.0, 2.0]}),
                sampling_frequency=90.0,
            )

    def test_trailing_zero_timestamps_do_not_create_invalid_interval(self, full_session):
        """Trailing zeros should not produce a start_time > end_time flag interval."""
        ts = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "timeSinceStartup": ts,
                "Node_Head_px": np.zeros(6),
            }
        )
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=df,
            sampling_frequency=90.0,
        )
        check = SamplingRateCheck()
        config = _make_config()

        flags = check(stream, full_session, config)
        rate_flags = [f for f in flags if "mismatch" in f.message.lower()]

        assert len(rate_flags) == 1
        assert rate_flags[0].start_time == pytest.approx(1.0)
        assert rate_flags[0].end_time == pytest.approx(3.0)
