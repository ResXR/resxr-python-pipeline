"""Tests for the data splitter (io/splitter.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from resxr.core.constants import TrackingSystem
from resxr.core.session import Session, SessionMetadata, TrackingStream
from resxr.io.splitter import is_system_enabled, split_continuous_data

# ===========================================================================
# Helper
# ===========================================================================

SAMPLING_FREQS = {
    "Head": 90.0,
    "Hands": 90.0,
    "Eyes": 90.0,
    "Face": 30.0,
    "Body": 30.0,
    "Controllers": 90.0,
}


def _make_session(metadata: SessionMetadata, continuous_df) -> Session:
    return Session(
        session_id="test",
        metadata=metadata,
        raw_continuous_data=continuous_df,
    )


# ===========================================================================
# is_system_enabled
# ===========================================================================


class TestIsSystemEnabled:
    def test_head_always_enabled(self, session_metadata):
        """HEAD is always enabled regardless of metadata flags."""
        session_metadata.hands_enabled = False
        assert is_system_enabled(TrackingSystem.HEAD, session_metadata) is True

    def test_hands_enabled_from_metadata(self, session_metadata):
        """HANDS is enabled when hands_enabled=True in metadata."""
        session_metadata.hands_enabled = True
        assert is_system_enabled(TrackingSystem.HANDS, session_metadata) is True

    def test_hands_disabled_from_metadata(self, session_metadata):
        """HANDS is disabled when hands_enabled=False in metadata."""
        session_metadata.hands_enabled = False
        assert is_system_enabled(TrackingSystem.HANDS, session_metadata) is False

    def test_eyes_enabled_from_metadata(self, session_metadata):
        session_metadata.eyes_enabled = True
        assert is_system_enabled(TrackingSystem.EYES, session_metadata) is True

    def test_eyes_disabled_from_metadata(self, session_metadata):
        session_metadata.eyes_enabled = False
        assert is_system_enabled(TrackingSystem.EYES, session_metadata) is False

    def test_face_enabled_from_metadata(self, session_metadata):
        session_metadata.face_enabled = True
        assert is_system_enabled(TrackingSystem.FACE, session_metadata) is True

    def test_body_disabled_from_metadata(self, session_metadata):
        session_metadata.body_enabled = False
        assert is_system_enabled(TrackingSystem.BODY, session_metadata) is False


# ===========================================================================
# split_continuous_data
# ===========================================================================


class TestSplitContinuousData:
    def test_returns_dict_of_tracking_streams(self, session_metadata, continuous_df):
        """split_continuous_data returns Dict[TrackingSystem, TrackingStream]."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, TrackingSystem)
            assert isinstance(val, TrackingStream)

    def test_head_stream_present(self, session_metadata, continuous_df):
        """HEAD stream is always created when HEAD columns are present."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert TrackingSystem.HEAD in result

    def test_head_stream_has_correct_columns(self, session_metadata, continuous_df):
        """HEAD stream data only contains HEAD-prefixed columns (+ timestamp)."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        head_cols = result[TrackingSystem.HEAD].data.columns.tolist()
        assert "timestamp" in head_cols
        # All non-time columns should be HEAD-prefixed
        data_cols = [c for c in head_cols if c != "timestamp"]
        assert all(c.startswith("Node_Head_") for c in data_cols)

    def test_hands_stream_has_correct_columns(self, session_metadata, continuous_df):
        """HANDS stream contains LeftHand_ and RightHand_ columns."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert TrackingSystem.HANDS in result
        hands_cols = result[TrackingSystem.HANDS].data.columns.tolist()
        data_cols = [c for c in hands_cols if c != "timestamp"]
        assert all(c.startswith("LeftHand_") or c.startswith("RightHand_") for c in data_cols)

    def test_face_not_in_result_from_continuous_data(self, session_metadata, continuous_df):
        """FACE is never included in continuous-data splitting (separate file)."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert TrackingSystem.FACE not in result

    def test_disabled_system_excluded_via_enabled_systems(self, session_metadata, continuous_df):
        """A system marked False in enabled_systems is excluded from results."""
        session = _make_session(session_metadata, continuous_df)
        enabled = {"Head": True, "Hands": False, "Eyes": True}
        result = split_continuous_data(session, enabled, SAMPLING_FREQS)
        assert TrackingSystem.HANDS not in result

    def test_system_disabled_via_metadata_excluded(self, session_metadata, continuous_df):
        """System not enabled in recording metadata is excluded even if columns exist."""
        session_metadata.hands_enabled = False
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert TrackingSystem.HANDS not in result

    def test_sampling_frequency_set_from_config(self, session_metadata, continuous_df):
        """Each stream's sampling_frequency matches the configured value."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert result[TrackingSystem.HEAD].sampling_frequency == pytest.approx(90.0)

    def test_timestamp_column_present_in_each_stream(self, session_metadata, continuous_df):
        """Every stream's DataFrame includes a 'timestamp' column."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        for stream in result.values():
            assert "timestamp" in stream.data.columns

    @pytest.mark.parametrize(
        "system", [TrackingSystem.HEAD, TrackingSystem.HANDS, TrackingSystem.EYES]
    )
    def test_stream_data_is_nonempty_dataframe(self, session_metadata, continuous_df, system):
        """Each created stream contains a non-empty DataFrame."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        if system in result:
            assert len(result[system].data) > 0

    def test_unknown_columns_not_assigned_to_any_stream(self, session_metadata, continuous_df):
        """Eyes_Closed columns (FACE-prefixed) are absent from non-FACE continuous streams."""
        session = _make_session(session_metadata, continuous_df)
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        for system, stream in result.items():
            data_cols = stream.data.columns.tolist()
            for col in data_cols:
                # Eyes_Closed_L/R should not appear in HEAD, HANDS, or EYES streams
                assert not col.startswith("Eyes_Closed"), (
                    f"Eyes_Closed column unexpectedly in {system.value} stream"
                )

    def test_empty_dataframe_returns_empty_dict(self, session_metadata):
        """Empty raw_continuous_data produces an empty result dict."""
        session = Session(
            session_id="test",
            metadata=session_metadata,
            raw_continuous_data=pd.DataFrame(),
        )
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert result == {}

    def test_none_dataframe_returns_empty_dict(self, session_metadata):
        """None raw_continuous_data produces an empty result dict."""
        session = Session(
            session_id="test",
            metadata=session_metadata,
            raw_continuous_data=None,
        )
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert result == {}

    def test_face_from_raw_face_data(self, session_metadata, continuous_df):
        """FACE stream is created from raw_face_data when present and face is enabled."""
        from tests.conftest import _face_df

        session = _make_session(session_metadata, continuous_df)
        session.raw_face_data = _face_df()
        result = split_continuous_data(session, None, SAMPLING_FREQS)
        assert TrackingSystem.FACE in result

    def test_missing_sampling_freq_for_enabled_system_raises(self, session_metadata, continuous_df):
        """Enabled system without a sampling_frequency entry raises ConfigurationError."""
        from resxr.core.exceptions import ConfigurationError

        session = _make_session(session_metadata, continuous_df)
        incomplete_freqs = {"Eyes": 90.0}  # missing Head, Hands
        with pytest.raises(ConfigurationError, match="sampling_frequencies"):
            split_continuous_data(session, None, incomplete_freqs)
