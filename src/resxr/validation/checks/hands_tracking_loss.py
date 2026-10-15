"""
Hand tracking loss detection check for ResXR pipeline.

Detects periods where hand tracking was lost for left and right hands separately using configurable flags.
Checks if configured columns (e.g., validity flags) indicate loss, and sets QualityFlag.group_name for reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.config import ValidationConfig
from ...core.constants import GLOBAL_CLOCK_COLUMN, TrackingSystem
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ...io.column_maps import get_hand_measurement_columns
from ...utils import preview
from ..registry import register_check

logger = get_logger(__name__)

# Text forms of a tracking-flag value (compared after strip and lower-casing)
_FLAG_TOKENS = {"1": 1.0, "1.0": 1.0, "true": 1.0, "0": 0.0, "0.0": 0.0, "false": 0.0}


def read_tracking_flag(values: pd.Series) -> tuple[pd.Series | None, list[str]]:
    """Read a tracking-flag column: 1/true = tracked, 0/false = lost, blank = lost.

    Returns ``(values, [])`` when every value is 0/1, true/false (any case) or
    blank; numeric and boolean columns are returned unchanged. Otherwise returns
    ``(None, unexpected)`` listing the other values, so no flags are made from a
    column the check cannot read.
    """
    present = values.dropna()
    if pd.api.types.is_numeric_dtype(values):  # includes bool
        unexpected = [str(v) for v in pd.unique(present) if v not in (0, 1)]
        return (None, unexpected) if unexpected else (values, [])

    mapping: dict = {}  # text column: each distinct value -> 1.0, 0.0 or NaN
    unexpected = []
    for value in pd.unique(present):
        token = str(value).strip().lower()
        if token in _FLAG_TOKENS:
            mapping[value] = _FLAG_TOKENS[token]
        elif not token:
            mapping[value] = np.nan
        else:
            unexpected.append(str(value))
    if unexpected:
        return None, unexpected
    return values.map(mapping).astype(float), []


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
    settings_keys = ("tracking_flags",)

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

        if GLOBAL_CLOCK_COLUMN not in df.columns:
            logger.error(
                f"Session '{session.session_id}': "
                f"{stream.system.value}: timeSinceStartup column missing — "
                "cannot create hands_tracking_loss flags. "
                "Check alternate_time_columns in pipeline_config.yaml."
            )
            session.record_check_not_run(
                self.name,
                stream.system.value,
                "no separate timeSinceStartup column (the stream has no per-system clock "
                "column; see preprocessing.alternate_time_columns)",
            )
            return flags
        ts_for_flags = df[GLOBAL_CLOCK_COLUMN].values

        for hand, columns in tracking_flags.items():
            for column in columns:
                if column not in df.columns:
                    logger.warning(
                        f"Session '{session.session_id}': hands_tracking_loss not run for "
                        f"{hand}: column {column} missing "
                        "(set validation.settings.tracking_flags)"
                    )
                    session.record_check_not_run(
                        self.name, stream.system.value, f"{hand}: column {column} missing"
                    )
                    continue
                tracked, unexpected = read_tracking_flag(df[column])
                if tracked is None:
                    logger.warning(
                        f"Session '{session.session_id}': hands_tracking_loss not run for "
                        f"{hand}: column {column} holds values other than 0/1, true/false "
                        f"or blank ({preview(unexpected)}); no hand-loss flags from this column"
                    )
                    session.record_check_not_run(
                        self.name,
                        stream.system.value,
                        f"{hand}: column {column} holds values other than 0/1, true/false "
                        f"or blank ({preview(unexpected)})",
                    )
                    continue
                tracking_lost = (tracked == 0) | (tracked.isna())
                if tracking_lost.any():
                    target_columns = get_hand_measurement_columns(df.columns.tolist(), column)
                    flags.extend(
                        QualityFlag.from_mask(
                            timestamps=ts_for_flags,
                            boolean_mask=tracking_lost.values,
                            check_name=self.name,
                            system=stream.system,
                            severity="warning",
                            message=f"Tracking loss detected: {hand} flag {column} indicates loss",
                            should_mask=bool(target_columns),
                            group_name=hand,
                            target_columns=target_columns,
                        )
                    )

        return flags


# Register the check
hands_tracking_loss_check = HandsTrackingLossCheck()
register_check(hands_tracking_loss_check)
