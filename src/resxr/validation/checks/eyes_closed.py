"""
Eyes closed detection check for ResXR pipeline.

Detects periods where both eyes are closed using the FACE tracking system.
"""

from __future__ import annotations

import numpy as np

from ...core.config import ValidationConfig
from ...core.constants import GLOBAL_CLOCK_COLUMN, TrackingSystem
from ...core.logger import get_logger
from ...core.session import QualityFlag, Session, TrackingStream
from ...io.column_maps import get_gaze_measurement_columns
from ..registry import has_global_clock, register_check

logger = get_logger(__name__)


class EyesClosedCheck:
    """
    Detect periods where both eyes are closed using the FACE stream.

    Checks if both Eyes_Closed_L and Eyes_Closed_R are above a threshold (default 0.9).
    This is a multistream check, so it can later be extended to affect other streams (e.g., EYES).
    """

    name = "eyes_closed"
    description = "Detects periods where both eyes are closed (FACE stream)"
    # Detection is FACE-based; the optional propagation to EYES is null-guarded
    # below, so only FACE is required. Listing EYES here would make the registry
    # skip the entire check on sessions without eye tracking.
    required_streams = [TrackingSystem.FACE]
    settings_keys = (
        "eyes_closed_threshold",
        "eyes_closed_min_duration",
        "eyes_closed_use_min_duration",
    )

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
        missing = [col for col in (col_left, col_right) if col not in df.columns]
        if missing:
            logger.warning(
                f"Session '{session.session_id}': eyes_closed not run: columns "
                f"{', '.join(missing)} missing from the face file (the check reads "
                f"{col_left} and {col_right})"
            )
            session.record_check_not_run(
                self.name, TrackingSystem.FACE.value, f"columns {', '.join(missing)} missing"
            )
            return flags
        if not has_global_clock(self.name, face_stream, session):
            return flags

        closed_segments: list[tuple] = []
        closed_mask = (df[col_left] >= threshold) & (df[col_right] >= threshold)
        if closed_mask.any():
            face_flags = QualityFlag.from_mask(
                timestamps=df[GLOBAL_CLOCK_COLUMN].values,
                boolean_mask=closed_mask.values,
                check_name=self.name,
                system=TrackingSystem.FACE,
                severity="info",
                message=f"Both eyes closed (threshold ≥ {threshold})",
                should_mask=False,
                group_name="both_eyes",
                target_columns=[col_left, col_right],
            )
            for f in face_flags:
                if (not use_min_duration) or (f.duration >= min_duration):
                    flags.append(f)
                    closed_segments.append((f.start_time, f.end_time))

        # Propagate mask to EYES stream for optional gaze filtering
        if eyes_stream is not None and not eyes_stream.data.empty and closed_segments:
            if GLOBAL_CLOCK_COLUMN not in eyes_stream.data.columns:
                logger.error(
                    f"Session '{session.session_id}': eyes_closed flags not carried over to "
                    "Eyes: no separate timeSinceStartup column in Eyes (the stream has no "
                    "per-system clock column; see preprocessing.alternate_time_columns)"
                )
                session.record_check_not_run(
                    self.name,
                    TrackingSystem.EYES.value,
                    "eyes-closed flags not carried over to Eyes: no separate timeSinceStartup "
                    "column in Eyes (see preprocessing.alternate_time_columns)",
                )
            else:
                eyes_ts = eyes_stream.data[GLOBAL_CLOCK_COLUMN].values
                eyes_closed_mask_for_eyes_stream = np.zeros_like(eyes_ts, dtype=bool)
                for seg_start, seg_end in closed_segments:
                    eyes_closed_mask_for_eyes_stream |= (eyes_ts >= seg_start) & (
                        eyes_ts <= seg_end
                    )
                if eyes_closed_mask_for_eyes_stream.any():
                    target_columns = get_gaze_measurement_columns(eyes_stream.data.columns.tolist())
                    flags.extend(
                        QualityFlag.from_mask(
                            timestamps=eyes_ts,
                            boolean_mask=eyes_closed_mask_for_eyes_stream,
                            check_name=self.name,
                            system=TrackingSystem.EYES,
                            severity="info",
                            message="Eyes closed (from FACE stream)",
                            should_mask=bool(target_columns),
                            group_name="both_eyes",
                            target_columns=target_columns,
                        )
                    )

        return flags


# Register the check
eyes_closed_check = EyesClosedCheck()
register_check(eyes_closed_check)
