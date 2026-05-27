"""
Hand tracking loss detection check for ResXR pipeline.

Detects periods where hand tracking was lost for left and right hands separately using configurable flags.
Checks if configured columns (e.g., validity flags) indicate loss, and sets QualityFlag.group_name for reporting.
"""

from __future__ import annotations

from ...core.config import ValidationConfig
from ...core.constants import TrackingSystem
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ..registry import register_check

logger = get_logger(__name__)


class HandsTrackingLossCheck:
    """
    Detect periods of tracking loss for left and right hands.

    For each hand, checks if configured columns (e.g., validity flags) indicate tracking loss.
    Uses the 'tracking_flags' configuration to determine which columns are affected.
    Sets QualityFlag.group_name to the hand ('left_hand', 'right_hand', etc.) for reporting and visualization.
    """

    name = "hands_tracking_loss"
    description = "Detects tracking loss for left and right hands using validity flags"
    required_streams = [TrackingSystem.HANDS]

    def __call__(
        self,
        stream: TrackingStream,
        session: Session,
        config: ValidationConfig,
    ) -> list[QualityFlag]:
        """Run hand tracking loss detection."""
        flags: list[QualityFlag] = []
        df = stream.data

        if df.empty or "timestamp" not in df.columns:
            return flags

        # Configurable flags for tracking loss detection
        tracking_flags = config.get(
            "tracking_flags",
            {
                "left_hand": ["LeftHand_Status_HandTracked"],
                "right_hand": ["RightHand_Status_HandTracked"],
            },
        )

        if "timeSinceStartup" not in df.columns:
            logger.error(
                f"{stream.system.value}: timeSinceStartup column missing — "
                "cannot create hands_tracking_loss flags. "
                "Check alternate_time_columns in pipeline_config.yaml."
            )
            return flags
        ts_for_flags = df["timeSinceStartup"].values

        for hand, columns in tracking_flags.items():
            for column in columns:
                if column in df.columns:
                    tracking_lost = (df[column] == 0) | (df[column].isna())
                    if tracking_lost.any():
                        flags.extend(
                            QualityFlag.from_mask(
                                timestamps=ts_for_flags,
                                boolean_mask=tracking_lost.values,
                                check_name=self.name,
                                system=stream.system,
                                severity="warning",
                                message=f"Tracking loss detected: {hand} flag {column} indicates loss",
                                should_mask=True,
                                group_name=hand,
                                target_columns=[column],
                            )
                        )

        return flags


# Register the check
hands_tracking_loss_check = HandsTrackingLossCheck()
register_check(hands_tracking_loss_check)
