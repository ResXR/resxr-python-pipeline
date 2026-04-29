"""Tests for the HTML report generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resxr.core.config import ReportConfig
from resxr.core.constants import TrackingSystem
from resxr.core.session import QualityFlag, Session, SessionMetadata, TrackingStream
from resxr.validation.checks.stats import compute_stream_stats
from resxr.visualization.report import ReportGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timestamps(n: int = 200, rate: float = 90.0, start: float = 1.0) -> np.ndarray:
    return np.arange(n, dtype=float) / rate + start


def _head_stream(n: int = 200) -> TrackingStream:
    rng = np.random.default_rng(42)
    ts = _make_timestamps(n, 90.0, 1.0)
    df = pd.DataFrame(
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
    stream = TrackingStream(system=TrackingSystem.HEAD, data=df, sampling_frequency=90.0)
    summary, detailed = compute_stream_stats(stream)
    stream.stats_summary = summary
    stream.stats_detailed = detailed
    return stream


def _minimal_session(streams: dict[TrackingSystem, TrackingStream] | None = None) -> Session:
    metadata = SessionMetadata(
        session_id="report_test_001",
        unity_version="2022.3.0f1",
        platform="Android",
        build_id="test_build",
        ovrplugin_version="60.0.0",
        sampling_mode="fixed",
        fixed_delta_time=0.011111,
        schema_rev="2.9",
        face_enabled=False,
        body_enabled=False,
        hands_enabled=False,
        eyes_enabled=False,
        controllers_enabled=False,
        detected_hand_bones=0,
        detected_body_joints=0,
    )
    if streams is None:
        streams = {TrackingSystem.HEAD: _head_stream()}
    return Session(
        session_id="report_test_001",
        subject_id="01",
        session_label="01",
        metadata=metadata,
        streams=streams,
    )


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    """Tests for ReportGenerator.generate()."""

    @pytest.fixture()
    def report_config(self, tmp_path) -> ReportConfig:
        return ReportConfig(enabled=True, output_dir=tmp_path)

    @pytest.fixture()
    def generator(self, report_config) -> ReportGenerator:
        return ReportGenerator(config=report_config)

    def test_generate_creates_html_file(self, generator, tmp_path):
        session = _minimal_session()
        result = generator.generate(session)
        assert result.exists()
        assert result.suffix == ".html"

    def test_generate_html_contains_session_id(self, generator, tmp_path):
        session = _minimal_session()
        result = generator.generate(session)
        content = result.read_text(encoding="utf-8")
        assert "report_test_001" in content

    def test_generate_respects_explicit_output_path(self, generator, tmp_path):
        session = _minimal_session()
        explicit = tmp_path / "custom_report.html"
        result = generator.generate(session, output_path=explicit)
        assert result == explicit
        assert explicit.exists()

    def test_generate_creates_parent_dirs(self, generator, tmp_path):
        session = _minimal_session()
        nested = tmp_path / "deep" / "nested" / "report.html"
        result = generator.generate(session, output_path=nested)
        assert result.exists()

    def test_generate_with_quality_flags(self, generator, tmp_path):
        stream = _head_stream()
        ts = stream.data["timestamp"].values
        stream.quality_flags = [
            QualityFlag(
                check_name="test_check",
                system=TrackingSystem.HEAD,
                start_time=float(ts[10]),
                end_time=float(ts[20]),
                severity="warning",
                message="Test warning flag",
                mask=False,
            )
        ]
        session = _minimal_session(streams={TrackingSystem.HEAD: stream})
        result = generator.generate(session)
        content = result.read_text(encoding="utf-8")
        assert result.exists()
        # The flag info should appear somewhere in the report
        assert "test_check" in content

    def test_generate_with_no_output_dir_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = ReportConfig(enabled=True, output_dir=None)
        gen = ReportGenerator(config=config)
        session = _minimal_session()
        result = gen.generate(session)
        assert result.resolve().exists()
        assert result.resolve().parent == tmp_path


class TestStreamStats:
    """Tests for _stream_stats helper."""

    @pytest.fixture()
    def generator(self, tmp_path) -> ReportGenerator:
        return ReportGenerator(config=ReportConfig(enabled=True, output_dir=tmp_path))

    def test_stream_stats_returns_list(self, generator):
        session = _minimal_session()
        stats = generator._stream_stats(session)
        assert isinstance(stats, list)
        assert len(stats) == 1

    def test_stream_stats_contains_expected_keys(self, generator):
        session = _minimal_session()
        stats = generator._stream_stats(session)
        entry = stats[0]
        assert entry["name"] == "Head"
        assert "rows" in entry
        assert "channels" in entry
        assert "nan_pct" in entry
        assert "detailed" in entry

    def test_stream_stats_skips_missing(self, generator):
        stream = _head_stream()
        stream.stats_summary = None
        stream.stats_detailed = None
        session = _minimal_session(streams={TrackingSystem.HEAD: stream})
        stats = generator._stream_stats(session)
        assert len(stats) == 0


class TestFlagsRelativeToOnset:
    """Tests for _flags_relative_to_onset."""

    def test_empty_flags_returns_empty(self):
        session = _minimal_session()
        flags = ReportGenerator._flags_relative_to_onset(session)
        assert flags == []

    def test_flag_times_are_relative_to_onset(self):
        stream = _head_stream()
        ts = stream.data["timestamp"].values
        onset = float(ts[0])
        stream.quality_flags = [
            QualityFlag(
                check_name="test",
                system=TrackingSystem.HEAD,
                start_time=float(ts[5]),
                end_time=float(ts[10]),
                severity="warning",
                message="flag",
                mask=False,
            )
        ]
        session = _minimal_session(streams={TrackingSystem.HEAD: stream})
        flags = ReportGenerator._flags_relative_to_onset(session)
        assert len(flags) == 1
        # Start time should be relative to onset (close to ts[5] - onset)
        expected_start = float(ts[5]) - onset
        assert abs(flags[0]["start"] - expected_start) < 0.01
