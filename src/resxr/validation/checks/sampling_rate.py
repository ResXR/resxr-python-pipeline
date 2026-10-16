"""
Sampling rate validation check for ResXR pipeline.

Validates that actual sampling rate matches expected rate, and that the
intervals between the stream's samples are regular.
"""

from __future__ import annotations

import numpy as np

from ...core.config import ValidationConfig
from ...core.constants import GLOBAL_CLOCK_COLUMN
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ...utils import find_first_nonzero_index, find_last_nonzero_index
from ..registry import has_global_clock, register_check

logger = get_logger(__name__)


class SamplingRateCheck:
    """
    Validate sampling frequency consistency.

    Flags the stream when its effective sampling rate deviates from the
    expected rate by more than ``sampling_rate_tolerance``, and when the
    intervals between its samples are irregular: their coefficient of
    variation exceeds ``sampling_cv_threshold``. Both use the stream's own
    clock (``timestamp``, its distinct finite non-zero values from the
    recording onset), where a new value marks a new SDK sample; rows are
    written in per-frame bursts on timeSinceStartup, so its intervals are not
    the samples'. The flags span the recording on timeSinceStartup.
    """

    name = "sampling_rate"
    description = "Validates sampling frequency consistency"
    required_streams = None
    settings_keys = ("sampling_rate_tolerance", "sampling_cv_threshold")

    def __call__(
        self, stream: TrackingStream, session: Session, config: ValidationConfig
    ) -> list[QualityFlag]:
        """Run sampling rate validation."""
        flags = []
        df = stream.data

        if len(df) < 2 or "timestamp" not in df.columns:
            return flags

        actual_rate = stream.sampling_frequency_effective
        expected_rate = stream.sampling_frequency

        if actual_rate <= 0:
            return flags

        if not has_global_clock(self.name, stream, session):
            return flags
        timestamps = df[GLOBAL_CLOCK_COLUMN].values

        onset_idx = find_first_nonzero_index(timestamps)
        offset_idx = find_last_nonzero_index(timestamps)
        if onset_idx is None or offset_idx is None or onset_idx > offset_idx:
            return flags

        # The flags span the recording window on timeSinceStartup
        valid_window_mask = np.zeros(len(timestamps), dtype=bool)
        valid_window_mask[onset_idx : offset_idx + 1] = True

        # Check deviation against tolerance from config
        if expected_rate > 0:
            deviation = abs(actual_rate - expected_rate) / expected_rate

            if deviation > config.get("sampling_rate_tolerance", 0.10):
                flags.extend(
                    QualityFlag.from_mask(
                        timestamps=timestamps,
                        boolean_mask=valid_window_mask,
                        severity="warning",
                        check_name=self.name,
                        system=stream.system,
                        message=(
                            f"Sampling rate mismatch: expected {expected_rate:.1f}Hz, "
                            f"got {actual_rate:.1f}Hz ({deviation * 100:.1f}% deviation)"
                        ),
                        should_mask=False,  # Don't mask for rate mismatch
                        target_columns=[],  # Sampling rate issues apply to all columns
                    )
                )

        # Irregular sampling: intervals between the stream's own samples (distinct
        # finite non-zero clock values from the recording onset, as for the rate)
        clock = df["timestamp"].to_numpy(dtype=float)
        first = find_first_nonzero_index(clock)
        time_diffs = np.array([])
        if first is not None:
            samples = np.unique(clock[np.isfinite(clock) & (clock != 0)])
            time_diffs = np.diff(samples[samples >= clock[first]])

        # Calculate coefficient of variation
        if len(time_diffs) > 1 and np.mean(time_diffs) > 0:
            cv = np.std(time_diffs) / np.mean(time_diffs)
            if cv > config.get("sampling_cv_threshold", 0.50):
                flags.extend(
                    QualityFlag.from_mask(
                        timestamps=timestamps,
                        boolean_mask=valid_window_mask,
                        severity="warning",
                        check_name=self.name,
                        system=stream.system,
                        message=f"Highly irregular sampling: CV={cv:.2f}",
                        should_mask=False,
                        target_columns=[],  # Irregular sampling applies to all columns
                    )
                )

        return flags


# Register the check
sampling_rate_check = SamplingRateCheck()
register_check(sampling_rate_check)
