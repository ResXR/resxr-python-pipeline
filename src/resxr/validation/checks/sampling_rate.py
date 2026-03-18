"""
Sampling rate validation check for ResXR pipeline.

Validates that actual sampling rate matches expected rate.
"""

from __future__ import annotations

import numpy as np

from ...core.config import ValidationConfig
from ...core.session import QualityFlag, Session, TrackingStream
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

        start = stream._start_timestamp()
        if start is None:
            return flags
        end = float(timestamps[-1])

        # Check deviation against tolerance from config
        if expected_rate > 0:
            deviation = abs(actual_rate - expected_rate) / expected_rate

            if deviation > config.get("sampling_rate_tolerance", 0.10):
                flags.append(
                    QualityFlag(
                        check_name=self.name,
                        system=stream.system,
                        start_time=start,
                        end_time=end,
                        severity="warning",
                        message=(
                            f"Sampling rate mismatch: expected {expected_rate:.1f}Hz, "
                            f"got {actual_rate:.1f}Hz ({deviation * 100:.1f}% deviation)"
                        ),
                        mask=False,  # Don't mask for rate mismatch
                        target_columns=[],  # Sampling rate issues apply to all columns
                    )
                )

        # Also check for highly irregular sampling
        unique_ts = np.unique(timestamps)
        unique_ts = unique_ts[unique_ts >= start]
        time_diffs = np.diff(unique_ts)

        # Calculate coefficient of variation
        if len(time_diffs) > 1 and np.mean(time_diffs) > 0:
            cv = np.std(time_diffs) / np.mean(time_diffs)
            if cv > config.get("sampling_cv_threshold", 0.50):
                flags.append(
                    QualityFlag(
                        check_name=self.name,
                        system=stream.system,
                        start_time=start,
                        end_time=end,
                        severity="warning",
                        message=f"Highly irregular sampling: CV={cv:.2f}",
                        mask=False,
                        target_columns=[],  # Irregular sampling applies to all columns
                    )
                )

        return flags


# Register the check
sampling_rate_check = SamplingRateCheck()
register_check(sampling_rate_check)
