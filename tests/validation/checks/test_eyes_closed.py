"""Tests for EyesClosedCheck (validation/checks/eyes_closed.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.core.config import ValidationConfig
from resxr.core.constants import TrackingSystem
from resxr.core.session import Session, SessionMetadata, TrackingStream
from resxr.validation.checks.eyes_closed import EyesClosedCheck
from tests.conftest import make_timestamps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**settings) -> ValidationConfig:
    return ValidationConfig(enabled_checks=["eyes_closed"], settings=settings)


def _make_face_stream(
    n: int = 100,
    left_values: np.ndarray | None = None,
    right_values: np.ndarray | None = None,
    rate: float = 30.0,
) -> TrackingStream:
    """FACE stream with configurable Eyes_Closed_L/R values."""
    ts = make_timestamps(n, rate, 1.0)
    if left_values is None:
        left_values = np.zeros(n)
    if right_values is None:
        right_values = np.zeros(n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "Eyes_Closed_L": left_values,
            "Eyes_Closed_R": right_values,
            "Jaw_Drop": np.zeros(n),
        }
    )
    return TrackingStream(
        system=TrackingSystem.FACE,
        data=df,
        sampling_frequency=rate,
    )


def _make_eyes_stream(n: int = 100, rate: float = 90.0) -> TrackingStream:
    """Minimal EYES stream."""
    ts = make_timestamps(n, rate, 1.0)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "LeftEye_px": np.zeros(n),
            "LeftEye_py": np.zeros(n),
        }
    )
    return TrackingStream(
        system=TrackingSystem.EYES,
        data=df,
        sampling_frequency=rate,
    )


def _make_session_with_face_and_eyes(
    face_stream: TrackingStream,
    eyes_stream: TrackingStream,
    metadata: SessionMetadata,
) -> Session:
    return Session(
        session_id="test",
        subject_id="01",
        session_label="01",
        metadata=metadata,
        streams={
            TrackingSystem.FACE: face_stream,
            TrackingSystem.EYES: eyes_stream,
        },
    )


def _make_session_face_only(
    face_stream: TrackingStream,
    metadata: SessionMetadata,
) -> Session:
    return Session(
        session_id="test",
        subject_id="01",
        session_label="01",
        metadata=metadata,
        streams={TrackingSystem.FACE: face_stream},
    )


# ===========================================================================
# Metadata
# ===========================================================================


class TestEyesClosedMetadata:
    def test_name(self):
        check = EyesClosedCheck()
        assert check.name == "eyes_closed"

    def test_description_is_string(self):
        check = EyesClosedCheck()
        assert isinstance(check.description, str)
        assert len(check.description) > 0

    def test_required_streams_includes_face_and_eyes(self):
        check = EyesClosedCheck()
        assert TrackingSystem.FACE in check.required_streams
        assert TrackingSystem.EYES in check.required_streams

    def test_required_streams_first_is_face(self):
        """run_all triggers the check when stream==required_streams[0]==FACE."""
        check = EyesClosedCheck()
        assert check.required_streams[0] == TrackingSystem.FACE


# ===========================================================================
# No flags when eyes are open
# ===========================================================================


class TestEyesClosedNoFlags:
    def test_eyes_open_no_flags(self, session_metadata):
        """Eyes fully open (values=0.0) → no flags."""
        face = _make_face_stream(n=100)
        eyes = _make_eyes_stream(n=100)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config()
        flags = check(face, session, config)
        assert flags == []

    def test_only_one_eye_closed_no_flag(self, session_metadata):
        """Only left eye closed → does not trigger (both must be closed)."""
        n = 100
        left = np.full(n, 0.95)  # above threshold
        right = np.full(n, 0.05)  # below threshold
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config()
        flags = check(face, session, config)
        assert flags == []

    def test_missing_face_stream_no_crash(self, session_metadata):
        """Session without FACE stream returns empty list without raising."""
        eyes = _make_eyes_stream(n=100)
        # Build a session whose FACE stream is absent
        session = Session(
            session_id="test",
            subject_id="01",
            session_label="01",
            metadata=session_metadata,
            streams={TrackingSystem.EYES: eyes},
        )
        check = EyesClosedCheck()
        config = _make_config()
        # We still call with an eyes stream as the trigger stream
        flags = check(eyes, session, config)
        assert flags == []


# ===========================================================================
# Flags when both eyes closed
# ===========================================================================


class TestEyesClosedFlagsGenerated:
    def test_both_eyes_closed_above_threshold_produces_flags(self, session_metadata):
        """Both eyes ≥ 0.9 (default threshold) → flags emitted."""
        n = 100
        left = np.full(n, 0.95)
        right = np.full(n, 0.95)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert len(face_flags) >= 1

    def test_flag_severity_is_info(self, session_metadata):
        """Eyes-closed flags have severity='info'."""
        n = 100
        left = right = np.full(n, 0.95)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        assert all(f.severity == "info" for f in flags)

    def test_flag_system_is_face(self, session_metadata):
        """FACE stream flags have system==TrackingSystem.FACE."""
        n = 100
        left = right = np.full(n, 0.95)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert len(face_flags) >= 1

    def test_flag_propagated_to_eyes_stream(self, session_metadata):
        """EYES stream also gets a flag when both eyes are closed."""
        n = 100
        left = right = np.full(n, 0.95)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        eyes_flags = [f for f in flags if f.system == TrackingSystem.EYES]
        assert len(eyes_flags) >= 1

    def test_no_eyes_stream_still_returns_face_flags(self, session_metadata):
        """When EYES stream is absent, only FACE flags are emitted (no crash)."""
        n = 100
        left = right = np.full(n, 0.95)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        session = _make_session_face_only(face, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert len(face_flags) >= 1
        eyes_flags = [f for f in flags if f.system == TrackingSystem.EYES]
        assert eyes_flags == []


# ===========================================================================
# Threshold and min_duration
# ===========================================================================


class TestEyesClosedThreshold:
    def test_default_threshold_is_0_9(self, session_metadata):
        """Values exactly at 0.9 trigger a flag (threshold is >= 0.9)."""
        n = 100
        left = right = np.full(n, 0.9)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert len(face_flags) >= 1

    def test_values_below_threshold_no_flag(self, session_metadata):
        """Values just below 0.9 do NOT trigger a flag."""
        n = 100
        left = right = np.full(n, 0.89)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(eyes_closed_use_min_duration=False)
        flags = check(face, session, config)
        assert flags == []

    @pytest.mark.parametrize("threshold", [0.5, 0.7])
    def test_configurable_threshold_triggers_flag(self, session_metadata, threshold):
        """Threshold below value (0.75) triggers flag."""
        n = 100
        left = right = np.full(n, 0.75)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(
            eyes_closed_threshold=threshold,
            eyes_closed_use_min_duration=False,
        )
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert len(face_flags) >= 1

    def test_configurable_threshold_above_value_no_flag(self, session_metadata):
        """Threshold above value (0.75) does not trigger flag."""
        n = 100
        left = right = np.full(n, 0.75)
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(
            eyes_closed_threshold=0.95,
            eyes_closed_use_min_duration=False,
        )
        flags = check(face, session, config)
        assert flags == []


class TestEyesClosedMinDuration:
    def test_min_duration_filters_short_segments(self, session_metadata):
        """Segments shorter than min_duration are filtered out."""
        # 30 Hz → 1 sample = 0.033s; min_duration=0.5s means single-sample segments filtered
        n = 200
        left = np.full(n, 0.1)
        right = np.full(n, 0.1)
        # Only rows 10–11 are closed (2 samples ≈ 0.067s at 30 Hz)
        left[10:12] = 0.95
        right[10:12] = 0.95
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(
            eyes_closed_use_min_duration=True,
            eyes_closed_min_duration=0.5,  # 500ms — far longer than 2-sample segment
        )
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert face_flags == []

    def test_min_duration_allows_long_segments(self, session_metadata):
        """Segments longer than min_duration are kept."""
        n = 200
        left = np.full(n, 0.1)
        right = np.full(n, 0.1)
        # Rows 10–60 closed (~1.67s at 30 Hz)
        left[10:60] = 0.95
        right[10:60] = 0.95
        face = _make_face_stream(n=n, left_values=left, right_values=right)
        eyes = _make_eyes_stream(n=n)
        session = _make_session_with_face_and_eyes(face, eyes, session_metadata)
        check = EyesClosedCheck()
        config = _make_config(
            eyes_closed_use_min_duration=True,
            eyes_closed_min_duration=0.5,
        )
        flags = check(face, session, config)
        face_flags = [f for f in flags if f.system == TrackingSystem.FACE]
        assert len(face_flags) >= 1
