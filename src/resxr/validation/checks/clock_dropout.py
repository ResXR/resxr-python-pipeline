"""
Clock dropout detection check for ResXR pipeline.

Detects contiguous zero/NaN blocks in the per-system timestamp column
(``timestamp``) that fall strictly inside the recording window (between
onset and offset).  Leading and trailing zeros — emitted during device
spin-up/spin-down — are excluded; only mid-recording dropouts are flagged.

During a clock dropout the per-system clock resets to 0 while
``timeSinceStartup`` (the global Unity clock) keeps ticking.  Affected
rows produce invalid ``latency`` values in BIDS output, so the check
sets ``mask=True`` and ``target_columns=[]`` to request full-row masking
for each dropout segment.

Flag boundaries are stored in ``timeSinceStartup`` units (consistent with
all other checks) so that ``apply_quality_masking`` can match them against
``data["timeSinceStartup"].values`` correctly.
"""

from __future__ import annotations

import numpy as np

from ...core.config import ValidationConfig
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ...utils import find_internal_zero_blocks
from ..registry import register_check

logger = get_logger(__name__)


class ClockDropoutCheck:
    """
    Detect per-system clock dropouts (zero timestamps) mid-recording.

    Runs independently on every enabled stream.  A dropout is a contiguous
    block of zero or NaN values in the ``timestamp`` column that appears
    between the recording onset and offset (leading/trailing zeros are
    ignored).

    Flag severity is ``"warning"``; ``mask=True`` requests NaN-replacement
    of all data columns in the affected rows during preprocessing.
    """

    name = "clock_dropout"
    description = (
        "Detects per-system clock dropouts (zero timestamps) strictly inside the recording window"
    )
    required_streams = None  # runs independently on every stream

    def __call__(
        self,
        stream: TrackingStream,
        session: Session,
        config: ValidationConfig,
    ) -> list[QualityFlag]:
        """Run clock dropout detection."""
        flags: list[QualityFlag] = []
        df = stream.data

        if df.empty or "timestamp" not in df.columns:
            return flags

        if "timeSinceStartup" not in df.columns:
            logger.error(
                "%s: timeSinceStartup column missing — cannot check for clock "
                "dropouts.  Check alternate_time_columns in pipeline_config.yaml.",
                stream.system.value,
            )
            return flags

        timestamps = df["timestamp"].values
        ts_global = df["timeSinceStartup"].values

        blocks = find_internal_zero_blocks(timestamps)
        if not blocks:
            return flags

        # Build a boolean mask over the global clock.
        # apply_quality_masking compares flag boundaries against timeSinceStartup,
        # so we pass ts_global as the timestamps argument to from_mask — consistent
        # with every other check in the codebase.
        n = len(ts_global)
        dropout_mask = np.zeros(n, dtype=bool)
        for start_idx, end_idx in blocks:
            dropout_mask[start_idx : end_idx + 1] = True

        flags.extend(
            QualityFlag.from_mask(
                timestamps=ts_global,
                boolean_mask=dropout_mask,
                check_name=self.name,
                system=stream.system,
                severity="warning",
                message="Per-system clock dropout: zero timestamp detected mid-recording",
                should_mask=True,
                target_columns=[],  # mask all data columns — entire row is timing-suspect
            )
        )
        return flags


# Register the check
clock_dropout_check = ClockDropoutCheck()
register_check(clock_dropout_check)
