"""
Eyes closed detection check for ResXR pipeline.

Detects periods where both eyes are closed using the FACE tracking system.
"""

from __future__ import annotations

import numpy as np

from ...core.config import ValidationConfig
from ...core.constants import TrackingSystem
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ..registry import register_check

logger = get_logger(__name__)


class EyesClosedCheck:
    """
    Detect periods where both eyes are closed using the FACE stream.

    Checks if both Eyes_Closed_L and Eyes_Closed_R are above a threshold (default 0.9).
    This is a multistream check, so it can later be extended to affect other streams (e.g., EYES).
    """

    name = "eyes_closed"
    description = "Detects periods where both eyes are closed (FACE stream)"
    # Require both FACE and EYES for multistream access
    required_streams = [TrackingSystem.FACE, TrackingSystem.EYES]

    def __call__(
        self,
        stream: TrackingStream,
        session: Session,
        config: ValidationConfig,
    ) -> list[QualityFlag]:
        """Run eyes closed detection as a multistream check."""
        flags: list[QualityFlag] = []

        # Access both streams
        face_stream = session.get_stream(TrackingSystem.FACE)
        eyes_stream = session.get_stream(TrackingSystem.EYES)

        if (
            face_stream is None
            or face_stream.data.empty
            or "timestamp" not in face_stream.data.columns
        ):
            return flags

        # Configurable thresholds - can be set in config/pipeline_config.yaml under validation.settings
        threshold = config.get("eyes_closed_threshold", 0.9)
        min_duration = config.get(
            "eyes_closed_min_duration", 0.1
        )  # seconds, typical blink ~0.1-0.15s
        use_min_duration = config.get("eyes_closed_use_min_duration", True)

        col_left = "Eyes_Closed_L"
        col_right = "Eyes_Closed_R"

        df = face_stream.data
        closed_segments: list[tuple] = []
        if col_left in df.columns and col_right in df.columns:
            closed_mask = (df[col_left] >= threshold) & (df[col_right] >= threshold)
            if closed_mask.any():
                if "timeSinceStartup" not in df.columns:
                    logger.error(
                        "FACE: timeSinceStartup column missing — "
                        "cannot create eyes_closed flags. "
                        "Check alternate_time_columns in pipeline_config.yaml."
                    )
                    return flags
                face_flags = QualityFlag.from_mask(
                    timestamps=df["timeSinceStartup"].values,
                    boolean_mask=closed_mask.values,
                    check_name=self.name,
                    system=TrackingSystem.FACE,
                    severity="info",
                    message=f"Both eyes closed (threshold ≥ {threshold})",
                    should_mask=True,
                    group_name="both_eyes",
                    target_columns=[col_left, col_right],
                )
                for f in face_flags:
                    if (not use_min_duration) or (f.duration >= min_duration):
                        flags.append(f)
                        closed_segments.append((f.start_time, f.end_time))

        # Propagate mask to EYES stream for optional gaze filtering
        if eyes_stream is not None and not eyes_stream.data.empty and closed_segments:
            if "timeSinceStartup" not in eyes_stream.data.columns:
                logger.error(
                    "EYES: timeSinceStartup column missing — "
                    "cannot propagate eyes_closed flags to Eyes stream. "
                    "Check alternate_time_columns in pipeline_config.yaml."
                )
            else:
                eyes_ts = eyes_stream.data["timeSinceStartup"].values
                eyes_closed_mask_for_eyes_stream = np.zeros_like(eyes_ts, dtype=bool)
                for seg_start, seg_end in closed_segments:
                    eyes_closed_mask_for_eyes_stream |= (eyes_ts >= seg_start) & (
                        eyes_ts <= seg_end
                    )
                if eyes_closed_mask_for_eyes_stream.any():
                    flags.extend(
                        QualityFlag.from_mask(
                            timestamps=eyes_ts,
                            boolean_mask=eyes_closed_mask_for_eyes_stream,
                            check_name=self.name,
                            system=TrackingSystem.EYES,
                            severity="info",
                            message="Eyes closed (from FACE stream)",
                            should_mask=True,
                            group_name="both_eyes",
                            target_columns=[],
                        )
                    )

        return flags


# Register the check
eyes_closed_check = EyesClosedCheck()
register_check(eyes_closed_check)
