"""Tests for utility functions (utils/__init__.py)."""

from __future__ import annotations

import numpy as np
import pytest

from resxr.utils import (
    find_first_nonzero_index,
    find_internal_zero_blocks,
    find_last_nonzero_index,
    find_recording_onset,
    format_duration,
)

# ===========================================================================
# format_duration
# ===========================================================================


class TestFormatDuration:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "0.0s"),
            (0.5, "0.5s"),
            (45.0, "45.0s"),
            (59.9, "59.9s"),
            (60.0, "1m 0s"),
            (90.0, "1m 30s"),
            (3599.0, "59m 59s"),
            (3600.0, "1h 0m 0s"),
            (3723.0, "1h 2m 3s"),
            (7261.0, "2h 1m 1s"),
        ],
    )
    def test_standard_cases(self, seconds, expected):
        """format_duration returns the expected string for known inputs."""
        assert format_duration(seconds) == expected

    def test_returns_string(self):
        """format_duration always returns a str."""
        assert isinstance(format_duration(100.0), str)

    def test_negative_seconds_returns_string(self):
        """Negative input returns a string (documents behaviour, no crash)."""
        result = format_duration(-5.0)
        assert isinstance(result, str)

    def test_fractional_seconds_below_60(self):
        """Fractional seconds < 60 include one decimal place."""
        result = format_duration(1.5)
        assert result == "1.5s"

    def test_large_value_includes_hours(self):
        """Value larger than 3600 produces an 'h' component."""
        result = format_duration(86400.0)
        assert "h" in result

    def test_one_minute_exactly(self):
        """Exactly 60 seconds switches from seconds format to minutes format."""
        result = format_duration(60.0)
        assert "m" in result
        assert "s" not in result.split("m")[0]  # 'h' not present


# ===========================================================================
# find_recording_onset
# ===========================================================================


class TestFindRecordingOnset:
    def test_normal_array_returns_first_nonzero(self):
        """[0, 0, 0, 1.0, 2.0] → 1.0."""
        ts = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert find_recording_onset(ts) == pytest.approx(1.0)

    def test_no_leading_zeros_returns_first_element(self):
        """Array without leading zeros → first element."""
        ts = np.array([1.0, 2.0, 3.0])
        assert find_recording_onset(ts) == pytest.approx(1.0)

    def test_all_zeros_returns_none(self):
        """All-zero array → None."""
        ts = np.zeros(10)
        assert find_recording_onset(ts) is None

    def test_empty_array_returns_none(self):
        """Empty array → None."""
        ts = np.array([])
        assert find_recording_onset(ts) is None

    def test_single_nonzero_element(self):
        """Single non-zero element → that value."""
        ts = np.array([42.0])
        assert find_recording_onset(ts) == pytest.approx(42.0)

    def test_single_zero_element_returns_none(self):
        """Single zero element → None."""
        ts = np.array([0.0])
        assert find_recording_onset(ts) is None

    def test_returns_float(self):
        """Return type is float (not numpy scalar that would break comparisons)."""
        ts = np.array([0.0, 5.0, 10.0])
        result = find_recording_onset(ts)
        assert isinstance(result, float)

    def test_large_array(self):
        """Works correctly on a large array with leading zeros."""
        ts = np.zeros(1000)
        ts[500] = 3.14
        ts[501:] = np.arange(1, 500, dtype=float)
        assert find_recording_onset(ts) == pytest.approx(3.14)

    def test_onset_is_correct_value(self):
        """Returns the value AT the first nonzero index, not the index itself."""
        ts = np.array([0.0, 0.0, 7.5, 8.0])
        assert find_recording_onset(ts) == pytest.approx(7.5)

    def test_negative_values_are_nonzero(self):
        """Negative timestamps count as non-zero."""
        ts = np.array([0.0, -1.0, 2.0])
        assert find_recording_onset(ts) == pytest.approx(-1.0)


# ===========================================================================
# find_last_nonzero_index
# ===========================================================================


class TestFindLastNonzeroIndex:
    def test_no_trailing_zeros_returns_last_index(self):
        """[1.0, 2.0, 3.0] → index 2 (the last element)."""
        ts = np.array([1.0, 2.0, 3.0])
        assert find_last_nonzero_index(ts) == 2

    def test_trailing_zeros_returns_last_nonzero(self):
        """[1.0, 2.0, 3.0, 0.0, 0.0] → index 2."""
        ts = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        assert find_last_nonzero_index(ts) == 2

    def test_leading_and_trailing_zeros(self):
        """[0, 0, 1.0, 2.0, 0, 0] → index 3."""
        ts = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
        assert find_last_nonzero_index(ts) == 3

    def test_all_zeros_returns_none(self):
        """All-zero array → None."""
        ts = np.zeros(5)
        assert find_last_nonzero_index(ts) is None

    def test_empty_array_returns_none(self):
        """Empty array → None."""
        ts = np.array([])
        assert find_last_nonzero_index(ts) is None

    def test_single_nonzero_element(self):
        """Single non-zero element → index 0."""
        ts = np.array([42.0])
        assert find_last_nonzero_index(ts) == 0

    def test_single_zero_element_returns_none(self):
        """Single zero element → None."""
        ts = np.array([0.0])
        assert find_last_nonzero_index(ts) is None

    def test_nan_trailing_values_skipped(self):
        """Trailing NaN values are skipped like trailing zeros."""
        ts = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        assert find_last_nonzero_index(ts) == 2

    def test_returns_int(self):
        """Return type is int (not numpy scalar)."""
        ts = np.array([0.0, 5.0, 10.0])
        result = find_last_nonzero_index(ts)
        assert isinstance(result, int)


# ===========================================================================
# find_first_nonzero_index
# ===========================================================================


class TestFindFirstNonzeroIndex:
    def test_no_leading_zeros_returns_zero(self):
        """[1.0, 2.0, 3.0] → index 0."""
        ts = np.array([1.0, 2.0, 3.0])
        assert find_first_nonzero_index(ts) == 0

    def test_leading_zeros_skipped(self):
        """[0, 0, 0, 1.0, 2.0] → index 3."""
        ts = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert find_first_nonzero_index(ts) == 3

    def test_all_zeros_returns_none(self):
        """All-zero array → None."""
        ts = np.zeros(10)
        assert find_first_nonzero_index(ts) is None

    def test_empty_array_returns_none(self):
        """Empty array → None."""
        ts = np.array([])
        assert find_first_nonzero_index(ts) is None

    def test_single_nonzero_returns_zero(self):
        """[5.0] → index 0."""
        ts = np.array([5.0])
        assert find_first_nonzero_index(ts) == 0

    def test_single_zero_returns_none(self):
        """[0.0] → None."""
        ts = np.array([0.0])
        assert find_first_nonzero_index(ts) is None

    def test_leading_nan_skipped(self):
        """NaN prefix is skipped — first non-zero finite value wins."""
        ts = np.array([np.nan, np.nan, 3.0, 4.0])
        assert find_first_nonzero_index(ts) == 2

    def test_negative_value_is_nonzero(self):
        """Negative timestamps count as non-zero."""
        ts = np.array([0.0, -1.0, 2.0])
        assert find_first_nonzero_index(ts) == 1

    def test_returns_int_type(self):
        """Return type is int (not numpy scalar)."""
        ts = np.array([0.0, 5.0, 10.0])
        result = find_first_nonzero_index(ts)
        assert isinstance(result, int)

    def test_large_array_with_leading_zeros(self):
        """Works on a large array; returns correct index."""
        ts = np.zeros(1000)
        ts[500] = 3.14
        assert find_first_nonzero_index(ts) == 500


# ===========================================================================
# find_internal_zero_blocks
# ===========================================================================


class TestFindInternalZeroBlocks:
    def test_all_valid_no_blocks(self):
        """No zeros or NaN inside window → []."""
        ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert find_internal_zero_blocks(ts) == []

    def test_empty_array_no_blocks(self):
        """Empty array → []."""
        assert find_internal_zero_blocks(np.array([])) == []

    def test_all_zeros_no_blocks(self):
        """All zeros → no valid onset/offset → []."""
        assert find_internal_zero_blocks(np.zeros(10)) == []

    def test_leading_zeros_excluded(self):
        """Leading zeros are not internal → []."""
        ts = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
        assert find_internal_zero_blocks(ts) == []

    def test_trailing_zeros_excluded(self):
        """Trailing zeros are not internal → []."""
        ts = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        assert find_internal_zero_blocks(ts) == []

    def test_single_internal_zero(self):
        """One internal zero → one (start, end) tuple."""
        ts = np.array([1.0, 0.0, 2.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 1)]

    def test_internal_block_of_three(self):
        """[1, 0, 0, 0, 2] → block covering indices 1–3."""
        ts = np.array([1.0, 0.0, 0.0, 0.0, 2.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 3)]

    def test_two_separate_internal_blocks(self):
        """Two disjoint zero blocks → two tuples in order."""
        ts = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 2), (4, 5)]

    def test_nan_mid_block_detected(self):
        """NaN inside window is treated as bad."""
        ts = np.array([1.0, np.nan, 2.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 1)]

    def test_inf_mid_block_detected(self):
        """inf is not finite → treated as bad."""
        ts = np.array([1.0, np.inf, 2.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 1)]

    def test_mixed_nan_zero_block_is_one_block(self):
        """Contiguous NaN and zero values form a single block."""
        ts = np.array([1.0, np.nan, 0.0, np.nan, 2.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 3)]

    def test_negative_values_not_flagged(self):
        """Negative timestamps are valid (nonzero + finite) — not flagged."""
        ts = np.array([1.0, -1.0, 0.0, 2.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(2, 2)]

    def test_single_valid_sample_no_interior(self):
        """[0, 5, 0] → onset == offset, no interior window → []."""
        ts = np.array([0.0, 5.0, 0.0])
        assert find_internal_zero_blocks(ts) == []

    def test_two_valid_samples_no_interior(self):
        """[1, 2] → window_start >= window_end → []."""
        ts = np.array([1.0, 2.0])
        assert find_internal_zero_blocks(ts) == []

    def test_block_at_onset_plus_one(self):
        """Zero immediately after onset is detected."""
        ts = np.array([1.0, 0.0, 2.0, 3.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 1)]

    def test_block_at_offset_minus_one(self):
        """Zero immediately before offset is detected."""
        ts = np.array([1.0, 2.0, 0.0, 3.0])
        result = find_internal_zero_blocks(ts)
        assert result == [(2, 2)]

    def test_integer_dtype_no_raise(self):
        """Integer dtype input works without raising."""
        ts = np.array([1, 0, 2], dtype=int)
        result = find_internal_zero_blocks(ts)
        assert result == [(1, 1)]

    def test_returns_list_of_tuples(self):
        """Return type is list[tuple[int, int]]."""
        ts = np.array([1.0, 0.0, 2.0])
        result = find_internal_zero_blocks(ts)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_each_block_start_le_end(self):
        """Every returned (start, end) has start <= end."""
        ts = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 3.0])
        for start, end in find_internal_zero_blocks(ts):
            assert start <= end
