"""Tests for StatsSummaryCheck (validation/checks/stats.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.core.config import ValidationConfig
from resxr.core.constants import TrackingSystem
from resxr.core.session import TrackingStream
from resxr.validation.checks.stats import (
    StatsSummaryCheck,
    compute_column_stats,
    compute_stream_stats,
)
from tests.conftest import make_timestamps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> ValidationConfig:
    return ValidationConfig(enabled_checks=["stats_summary"], settings={})


def _make_stream_with_nans(
    n: int = 100,
    nan_indices: list[int] | None = None,
) -> TrackingStream:
    ts = make_timestamps(n, 90.0, 1.0)
    values = np.ones(n, dtype=float)
    if nan_indices:
        for i in nan_indices:
            values[i] = np.nan
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "Node_Head_px": values,
            "Node_Head_py": values * 2,
        }
    )
    return TrackingStream(
        system=TrackingSystem.HEAD,
        data=df,
        sampling_frequency=90.0,
    )


# ===========================================================================
# Metadata
# ===========================================================================


class TestStatsSummaryMetadata:
    def test_name(self):
        check = StatsSummaryCheck()
        assert check.name == "stats_summary"

    def test_description_is_string(self):
        check = StatsSummaryCheck()
        assert isinstance(check.description, str)
        assert len(check.description) > 0


# ===========================================================================
# StatsSummaryCheck.__call__
# ===========================================================================


class TestStatsSummaryCheckCall:
    def test_returns_empty_list(self, head_stream, full_session):
        """StatsSummaryCheck never emits quality flags."""
        check = StatsSummaryCheck()
        config = _make_config()
        flags = check(head_stream, full_session, config)
        assert flags == []

    def test_sets_stats_summary_on_stream(self, head_stream, full_session):
        """stream.stats_summary is set after the check runs."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        assert hasattr(head_stream, "stats_summary")
        assert head_stream.stats_summary is not None

    def test_sets_stats_detailed_on_stream(self, head_stream, full_session):
        """stream.stats_detailed is set after the check runs."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        assert hasattr(head_stream, "stats_detailed")
        assert head_stream.stats_detailed is not None

    def test_stats_summary_is_dataframe(self, head_stream, full_session):
        """stats_summary is a pandas DataFrame."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        assert isinstance(head_stream.stats_summary, pd.DataFrame)

    def test_stats_detailed_is_dataframe(self, head_stream, full_session):
        """stats_detailed is a pandas DataFrame."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        assert isinstance(head_stream.stats_detailed, pd.DataFrame)

    def test_stats_detailed_has_entry_per_data_column(self, head_stream, full_session):
        """stats_detailed has one row per numeric non-timestamp column."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        # head_stream has: timestamp (excluded), Node_Head_px/py/pz/qw (4 cols)
        data_cols = [
            c
            for c in head_stream.data.columns
            if c not in {"timestamp", "timeSinceStartup", "Eyes_Time"}
            and pd.api.types.is_numeric_dtype(head_stream.data[c])
        ]
        assert len(head_stream.stats_detailed) == len(data_cols)

    def test_stats_detailed_contains_mean_std_nan_count(self, head_stream, full_session):
        """stats_detailed DataFrame has mean, std, nan_count columns."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        for col in ("mean", "std", "nan_count"):
            assert col in head_stream.stats_detailed.columns

    def test_stats_summary_has_row_count(self, head_stream, full_session):
        """stats_summary contains 'row_count' column."""
        check = StatsSummaryCheck()
        config = _make_config()
        check(head_stream, full_session, config)
        assert "row_count" in head_stream.stats_summary.columns

    def test_empty_stream_no_crash(self, full_session):
        """Empty DataFrame does not raise an exception."""
        stream = TrackingStream(
            system=TrackingSystem.HEAD,
            data=pd.DataFrame(columns=["timestamp", "Node_Head_px"]),
            sampling_frequency=90.0,
        )
        check = StatsSummaryCheck()
        config = _make_config()
        # Should not raise
        flags = check(stream, full_session, config)
        assert flags == []

    def test_nan_heavy_stream_no_crash(self, full_session):
        """Stream with many NaNs does not raise; nan_count is reflected."""
        stream = _make_stream_with_nans(n=100, nan_indices=list(range(50, 100)))
        check = StatsSummaryCheck()
        config = _make_config()
        flags = check(stream, full_session, config)
        assert flags == []
        # nan_count should be 50 for each column
        assert int(stream.stats_detailed.loc["Node_Head_px", "nan_count"]) == 50


# ===========================================================================
# compute_column_stats (unit-level)
# ===========================================================================


class TestComputeColumnStats:
    def test_returns_dict(self):
        """compute_column_stats returns a dict."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = compute_column_stats(series)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        """All expected stat keys are present in the result dict."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = compute_column_stats(series)
        for key in (
            "count",
            "nan_count",
            "nan_pct",
            "mean",
            "median",
            "std",
            "min",
            "p5",
            "p25",
            "p75",
            "p95",
            "max",
        ):
            assert key in result

    def test_mean_correct(self):
        """mean value is computed correctly."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = compute_column_stats(series)
        assert result["mean"] == pytest.approx(2.0)

    def test_nan_count_correct(self):
        """nan_count reflects the number of NaN values in the series."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan])
        result = compute_column_stats(series)
        assert result["nan_count"] == 2

    def test_all_nan_returns_nan_stats(self):
        """All-NaN series returns nan_pct=100 and NaN aggregates."""
        series = pd.Series([np.nan, np.nan])
        result = compute_column_stats(series)
        assert result["nan_pct"] == pytest.approx(100.0)
        assert np.isnan(result["mean"])


# ===========================================================================
# compute_stream_stats (unit-level)
# ===========================================================================


class TestComputeStreamStats:
    def test_returns_two_dataframes(self, head_stream):
        """compute_stream_stats returns a (summary, detailed) tuple of DataFrames."""
        summary, detailed = compute_stream_stats(head_stream)
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(detailed, pd.DataFrame)

    def test_summary_has_row_count_equal_to_stream_rows(self, head_stream):
        """summary['row_count'] matches the number of rows in the stream data."""
        summary, _ = compute_stream_stats(head_stream)
        assert int(summary["row_count"].iloc[0]) == len(head_stream.data)

    def test_detailed_excludes_timestamp_columns(self, head_stream):
        """'timestamp' is not included in the detailed stats index."""
        _, detailed = compute_stream_stats(head_stream)
        assert "timestamp" not in detailed.index
