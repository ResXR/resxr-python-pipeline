"""Shared pytest fixtures for the ResXR test suite.

All fixtures are function-scoped (default) so every test is fully isolated —
no shared mutable state between tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from resxr.core.config import (
    PipelineConfig,
    PreprocessingConfig,
    ValidationConfig,
)
from resxr.core.constants import TrackingSystem
from resxr.core.session import QualityFlag, Session, SessionMetadata, TrackingStream

# ---------------------------------------------------------------------------
# Timestamp helper (plain function — not a fixture)
# ---------------------------------------------------------------------------


def make_timestamps(n: int = 200, rate: float = 90.0, start: float = 1.0) -> np.ndarray:
    """Return a uniformly-spaced float64 timestamp array.

    Starting at *start* (non-zero by default) so find_recording_onset
    returns *start* immediately without needing to skip leading zeros.
    """
    return np.arange(n, dtype=float) / rate + start


# ---------------------------------------------------------------------------
# Private DataFrame factories (used inside fixtures below)
# ---------------------------------------------------------------------------


def _head_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = make_timestamps(n, 90.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "Node_Head_px": rng.normal(0.0, 0.1, n),
            "Node_Head_py": rng.normal(1.5, 0.05, n),
            "Node_Head_pz": rng.normal(0.0, 0.1, n),
            "Node_Head_qx": rng.uniform(-0.1, 0.1, n),
            "Node_Head_qy": rng.uniform(-0.1, 0.1, n),
            "Node_Head_qz": rng.uniform(-0.1, 0.1, n),
            "Node_Head_qw": rng.uniform(0.9, 1.0, n),
        }
    )


def _hands_df(n: int = 200, with_validity: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    ts = make_timestamps(n, 90.0, 1.0)
    data: dict = {
        "timestamp": ts,
        "timeSinceStartup": ts,
        "LeftHand_Root_px": rng.normal(0.3, 0.05, n),
        "LeftHand_Root_py": rng.normal(1.0, 0.05, n),
        "RightHand_Root_px": rng.normal(-0.3, 0.05, n),
        "RightHand_Root_py": rng.normal(1.0, 0.05, n),
    }
    if with_validity:
        data["LeftHand_Status_HandTracked"] = np.ones(n, dtype=int)
        data["RightHand_Status_HandTracked"] = np.ones(n, dtype=int)
    return pd.DataFrame(data)


def _eyes_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(45)
    ts = make_timestamps(n, 90.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "LeftEye_px": rng.normal(0.0, 0.01, n),
            "LeftEye_py": rng.normal(0.0, 0.01, n),
        }
    )


def _face_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    ts = make_timestamps(n, 30.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "timeSinceStartup": ts,
            "Eyes_Closed_L": rng.uniform(0.0, 0.1, n),
            "Eyes_Closed_R": rng.uniform(0.0, 0.1, n),
            "Jaw_Drop": rng.uniform(0.0, 0.3, n),
        }
    )


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def continuous_df() -> pd.DataFrame:
    """200-row DataFrame with HEAD, HANDS, EYES, and FACE-prefixed columns."""
    rng = np.random.default_rng(0)
    ts = make_timestamps(200, 90.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            # HEAD
            "Node_Head_px": rng.normal(0.0, 0.1, 200),
            "Node_Head_py": rng.normal(1.5, 0.05, 200),
            "Node_Head_pz": rng.normal(0.0, 0.1, 200),
            "Node_Head_qx": rng.uniform(-0.1, 0.1, 200),
            "Node_Head_qy": rng.uniform(-0.1, 0.1, 200),
            "Node_Head_qz": rng.uniform(-0.1, 0.1, 200),
            "Node_Head_qw": rng.uniform(0.9, 1.0, 200),
            # HANDS
            "LeftHand_Root_px": rng.normal(0.3, 0.05, 200),
            "RightHand_Root_px": rng.normal(-0.3, 0.05, 200),
            "LeftHand_Status_HandTracked": np.ones(200, dtype=int),
            "RightHand_Status_HandTracked": np.ones(200, dtype=int),
            # EYES
            "LeftEye_px": rng.normal(0.0, 0.01, 200),
            "LeftEye_py": rng.normal(0.0, 0.01, 200),
            # FACE blend shapes (skipped by splitter from continuous data)
            "Eyes_Closed_L": rng.uniform(0.0, 0.05, 200),
            "Eyes_Closed_R": rng.uniform(0.0, 0.05, 200),
        }
    )


@pytest.fixture
def face_df() -> pd.DataFrame:
    """200-row FACE stream DataFrame at 30 Hz."""
    return _face_df(200)


@pytest.fixture
def hands_df_with_validity() -> pd.DataFrame:
    """200-row HANDS DataFrame with validity columns (all 1 = tracking valid)."""
    return _hands_df(200, with_validity=True)


# ---------------------------------------------------------------------------
# QualityFlag fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_quality_flag():
    """Factory fixture: returns a callable that builds QualityFlags with sensible defaults."""

    def _factory(**kwargs) -> QualityFlag:
        defaults = {
            "check_name": "test_check",
            "system": TrackingSystem.HEAD,
            "start_time": 1.0,
            "end_time": 2.0,
            "severity": "warning",
            "message": "Test flag",
            "mask": False,
            "group_name": None,
            "target_columns": [],
        }
        defaults.update(kwargs)
        return QualityFlag(**defaults)

    return _factory


@pytest.fixture
def sample_flag_warning(make_quality_flag) -> QualityFlag:
    """Warning-severity HEAD flag with mask=False."""
    return make_quality_flag()


@pytest.fixture
def sample_flag_masking() -> QualityFlag:
    """Error-severity HANDS masking flag spanning rows 10–20 of a 90 Hz stream."""
    ts = make_timestamps(200, 90.0, 1.0)
    return QualityFlag(
        check_name="masking_check",
        system=TrackingSystem.HANDS,
        start_time=float(ts[10]),
        end_time=float(ts[20]),
        severity="error",
        message="Masking error",
        mask=True,
    )


# ---------------------------------------------------------------------------
# TrackingStream fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def head_stream() -> TrackingStream:
    """HEAD TrackingStream — 200 rows at 90 Hz, no quality flags."""
    return TrackingStream(
        system=TrackingSystem.HEAD,
        data=_head_df(),
        sampling_frequency=90.0,
    )


@pytest.fixture
def hands_stream() -> TrackingStream:
    """HANDS TrackingStream with validity columns (all valid)."""
    return TrackingStream(
        system=TrackingSystem.HANDS,
        data=_hands_df(),
        sampling_frequency=90.0,
    )


@pytest.fixture
def eyes_stream() -> TrackingStream:
    """EYES TrackingStream — 200 rows at 90 Hz."""
    return TrackingStream(
        system=TrackingSystem.EYES,
        data=_eyes_df(),
        sampling_frequency=90.0,
    )


@pytest.fixture
def face_stream(face_df) -> TrackingStream:
    """FACE TrackingStream — 200 rows at 30 Hz."""
    return TrackingStream(
        system=TrackingSystem.FACE,
        data=face_df,
        sampling_frequency=30.0,
    )


@pytest.fixture
def stream_with_flags() -> TrackingStream:
    """HEAD stream pre-loaded with 2 warning flags (mask=False) and 1 error flag (mask=True)."""
    stream = TrackingStream(
        system=TrackingSystem.HEAD,
        data=_head_df(),
        sampling_frequency=90.0,
    )
    ts = make_timestamps(200, 90.0, 1.0)
    stream.quality_flags = [
        QualityFlag(
            check_name="check_a",
            system=TrackingSystem.HEAD,
            start_time=float(ts[5]),
            end_time=float(ts[10]),
            severity="warning",
            message="Warning A",
            mask=False,
        ),
        QualityFlag(
            check_name="check_a",
            system=TrackingSystem.HEAD,
            start_time=float(ts[20]),
            end_time=float(ts[30]),
            severity="warning",
            message="Warning B",
            mask=False,
        ),
        QualityFlag(
            check_name="masking_check",
            system=TrackingSystem.HEAD,
            start_time=float(ts[50]),
            end_time=float(ts[60]),
            severity="error",
            message="Masking error",
            mask=True,
        ),
    ]
    return stream


# ---------------------------------------------------------------------------
# SessionMetadata fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def session_metadata() -> SessionMetadata:
    """SessionMetadata with HEAD, HANDS, EYES, FACE enabled; BODY and CONTROLLERS disabled."""
    return SessionMetadata(
        session_id="test_session_001",
        unity_version="2022.3.0f1",
        platform="Android",
        build_id="test_build",
        ovrplugin_version="60.0.0",
        sampling_mode="fixed",
        fixed_delta_time=0.011111,
        schema_rev="2.9",
        face_enabled=True,
        body_enabled=False,
        hands_enabled=True,
        eyes_enabled=True,
        controllers_enabled=False,
        detected_hand_bones=24,
        detected_body_joints=0,
    )


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_session(session_metadata) -> Session:
    """Session containing only a HEAD stream (200 rows at 90 Hz)."""
    return Session(
        session_id="test_session_001",
        subject_id="01",
        session_label="01",
        metadata=session_metadata,
        streams={
            TrackingSystem.HEAD: TrackingStream(
                system=TrackingSystem.HEAD,
                data=_head_df(),
                sampling_frequency=90.0,
            )
        },
    )


@pytest.fixture
def full_session(session_metadata) -> Session:
    """Session with HEAD, HANDS, EYES (90 Hz) and FACE (30 Hz) streams — 200 rows each."""
    return Session(
        session_id="test_session_001",
        subject_id="01",
        session_label="01",
        metadata=session_metadata,
        streams={
            TrackingSystem.HEAD: TrackingStream(
                system=TrackingSystem.HEAD,
                data=_head_df(),
                sampling_frequency=90.0,
            ),
            TrackingSystem.HANDS: TrackingStream(
                system=TrackingSystem.HANDS,
                data=_hands_df(),
                sampling_frequency=90.0,
            ),
            TrackingSystem.EYES: TrackingStream(
                system=TrackingSystem.EYES,
                data=_eyes_df(),
                sampling_frequency=90.0,
            ),
            TrackingSystem.FACE: TrackingStream(
                system=TrackingSystem.FACE,
                data=_face_df(),
                sampling_frequency=30.0,
            ),
        },
    )


@pytest.fixture
def session_with_flags(full_session) -> Session:
    """full_session where each stream has one warning flag (mask=False)."""
    ts = make_timestamps(200, 90.0, 1.0)
    for system, stream in full_session.streams.items():
        stream.quality_flags = [
            QualityFlag(
                check_name="test_check",
                system=system,
                start_time=float(ts[10]),
                end_time=float(ts[20]),
                severity="warning",
                message=f"{system.value} test flag",
                mask=False,
            )
        ]
    return full_session


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config_dict() -> dict:
    """Valid minimal pipeline config as a plain Python dict.

    All path strings are placeholders — they do not need to exist on disk
    for PipelineConfig.from_yaml() to succeed.
    """
    return {
        "bids": {
            "missing_values": "n/a",
            "dataset_type": "raw",
            "license": "CC0",
            "authors": ["Test Author"],
            "reference_frame": {
                "description": "Right-hand coordinate system",
                "rotation_rule": "right-hand",
                "rotation_order": "XYZ",
                "spatial_axes": "RAS",
            },
        },
        "sampling_frequencies": {
            "Head": 90.0,
            "Hands": 90.0,
            "Eyes": 90.0,
            "Face": 30.0,
            "Body": 30.0,
            "Controllers": 90.0,
        },
        "system_descriptions": {
            "Head": "Head tracking from VR headset",
            "Hands": "Hand tracking",
            "Eyes": "Eye tracking",
            "Face": "Face tracking",
        },
        "input": {
            "data_dir": "DATA/test_sessions",
            "continuous_data_pattern": "*_ContinuousData.csv",
            "face_data_pattern": "*_FaceExpressionData.csv",
            "metadata_pattern": "session_metadata.json",
            "events_data_pattern": "*_EventsData.csv",
        },
        "output": {
            "bids_root": "output/bids",
            "dataset_name": "TestDataset",
            "bids_version": "1.10.1",
            "task_name": "vr",
            "overwrite": True,
        },
        "device": {
            "manufacturer": "Meta",
            "model_name": "Meta Quest Pro",
        },
        "validation": {
            "enabled_checks": ["sampling_rate", "stats_summary"],
        },
        "preprocessing": {
            "apply_quality_masking": False,
        },
        "report": {
            "enabled": False,
            "output_dir": None,
        },
    }


@pytest.fixture
def tmp_config_yaml(tmp_path, minimal_config_dict) -> Path:
    """Write minimal_config_dict to a temporary YAML file; return the path."""
    path = tmp_path / "pipeline_config.yaml"
    with open(path, "w") as f:
        yaml.dump(minimal_config_dict, f)
    return path


@pytest.fixture
def pipeline_config(tmp_config_yaml) -> PipelineConfig:
    """PipelineConfig loaded from the temporary YAML fixture."""
    return PipelineConfig.from_yaml(tmp_config_yaml)


@pytest.fixture
def validation_config(pipeline_config) -> ValidationConfig:
    """ValidationConfig slice from pipeline_config."""
    return pipeline_config.validation


@pytest.fixture
def preprocessing_config(pipeline_config) -> PreprocessingConfig:
    """PreprocessingConfig slice from pipeline_config."""
    return pipeline_config.preprocessing


# ---------------------------------------------------------------------------
# File fixtures  (all use tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_continuous_csv(tmp_path) -> Path:
    """ContinuousData CSV with 'timeSinceStartup' column (not yet renamed to 'timestamp')."""
    ts = make_timestamps(20, 90.0, 1.0)
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        {
            "timeSinceStartup": ts,
            "Node_Head_px": rng.normal(0.0, 0.1, 20),
            "Node_Head_py": rng.normal(1.5, 0.05, 20),
            "LeftHand_Root_px": rng.normal(0.3, 0.05, 20),
        }
    )
    path = tmp_path / "test_ContinuousData.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_face_csv(tmp_path) -> Path:
    """FaceExpressionData CSV with a 'timestamp' column."""
    ts = make_timestamps(20, 30.0, 1.0)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "Eyes_Closed_L": np.zeros(20),
            "Eyes_Closed_R": np.zeros(20),
            "Jaw_Drop": np.zeros(20),
        }
    )
    path = tmp_path / "test_FaceExpressionData.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_events_csv(tmp_path) -> Path:
    """Events CSV with 'name', 'onset', 'duration' columns."""
    df = pd.DataFrame(
        {
            "name": ["start", "stimulus_1", "end"],
            "onset": [0.0, 5.0, 10.0],
            "duration": [0.0, 2.0, 0.0],
        }
    )
    path = tmp_path / "test_EventsData.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_metadata_json(tmp_path) -> Path:
    """Minimal session_metadata.json with known values."""
    metadata = {
        "session_id": "test_session_001",
        "utc_start_iso8601": "2024-01-01T12:00:00",
        "device_utc_offset": "+00:00",
        "unity_version": "2022.3.0f1",
        "platform": "Android",
        "build_id": "test_build",
        "ovrplugin_runtime_version": "60.0.0",
        "sampling_mode": "fixed",
        "fixedDeltaTime": 0.011111,
        "schema_rev": "2.9",
        "face_enabled": True,
        "body_enabled": False,
        "hands_enabled": True,
        "eyes_enabled": True,
        "controllers_enabled": False,
        "detected_hand_bones": 24,
        "detected_body_joints": 0,
    }
    path = tmp_path / "session_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f)
    return path


@pytest.fixture
def tmp_session_dir(
    tmp_path, tmp_continuous_csv, tmp_face_csv, tmp_events_csv, tmp_metadata_json
) -> Path:
    """Session directory containing all four required data files."""
    # All file fixtures already write into tmp_path — just return it.
    return tmp_path
