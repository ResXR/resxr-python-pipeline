"""
Sampling rate validation check for ResXR pipeline.

Validates that actual sampling rate matches expected rate.
"""

from __future__ import annotations

import numpy as np

from ...core.config import ValidationConfig
from ...core.session import QualityFlag, Session, TrackingStream
from ...utils import find_recording_offset_index, find_recording_onset_index
from ..registry import register_check


class SamplingRateCheck:
    """
    Validate sampling frequency consistency.

    Calculates the effective sampling rate and flags significant deviations
    from the expected rate.
    """

    name = "sampling_rate"
    description = "Validates sampling frequency consistency"
    required_streams = None

    def __call__(
        self, stream: TrackingStream, session: Session, config: ValidationConfig
    ) -> list[QualityFlag]:
        """Run sampling rate validation."""
        flags = []
        df = stream.data

        if len(df) < 2 or "timestamp" not in df.columns:
            return flags

        timestamps = df["timestamp"].values
        actual_rate = stream.sampling_frequency_effective
        expected_rate = stream.sampling_frequency

        if actual_rate <= 0:
            return flags

        onset_idx = find_recording_onset_index(timestamps)
        offset_idx = find_recording_offset_index(timestamps)
        if onset_idx is None or offset_idx is None or onset_idx > offset_idx:
            return flags

        # Use a slice for numeric work and a full-length mask for from_mask().
        window = slice(onset_idx, offset_idx + 1)
        valid_window_mask = np.zeros(len(timestamps), dtype=bool)
        valid_window_mask[window] = True
        valid_timestamps = timestamps[window]

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

        # Also check for highly irregular sampling
        unique_ts = np.unique(valid_timestamps[np.isfinite(valid_timestamps)])
        time_diffs = np.diff(unique_ts)

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
