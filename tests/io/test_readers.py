"""Tests for I/O reader functions (io/readers.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from resxr.core.config import InputConfig
from resxr.core.exceptions import DataLoadError, MissingDataError
from resxr.core.session import Session, SessionMetadata
from resxr.io.readers import (
    discover_sessions,
    load_continuous_data,
    load_events_data,
    load_face_data,
    load_session,
    load_session_metadata,
)

# ===========================================================================
# load_continuous_data
# ===========================================================================


class TestLoadContinuousData:
    def test_returns_dataframe(self, tmp_continuous_csv):
        """load_continuous_data returns a pandas DataFrame."""
        df = load_continuous_data(tmp_continuous_csv)
        assert isinstance(df, pd.DataFrame)

    def test_renames_timeSinceStartup_to_timestamp(self, tmp_continuous_csv):
        """'timeSinceStartup' column is renamed to 'timestamp'."""
        df = load_continuous_data(tmp_continuous_csv)
        assert "timestamp" in df.columns
        assert "timeSinceStartup" not in df.columns

    def test_nonzero_row_count(self, tmp_continuous_csv):
        """Loaded DataFrame has rows."""
        df = load_continuous_data(tmp_continuous_csv)
        assert len(df) > 0

    def test_missing_file_raises_data_load_error(self, tmp_path):
        """Nonexistent path raises DataLoadError."""
        with pytest.raises(DataLoadError):
            load_continuous_data(tmp_path / "nonexistent.csv")

    def test_csv_without_timestamp_column_raises(self, tmp_path):
        """CSV with no recognisable timestamp column raises DataLoadError."""
        no_ts = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(no_ts, index=False)
        with pytest.raises(DataLoadError):
            load_continuous_data(no_ts)

    def test_empty_csv_no_columns_raises(self, tmp_path):
        """Completely empty CSV (no columns at all) raises DataLoadError."""
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises(DataLoadError):
            load_continuous_data(empty)

    def test_empty_csv_with_header_returns_empty_df(self, tmp_path):
        """CSV with 'timeSinceStartup' header but zero rows returns empty DataFrame."""
        empty_with_header = tmp_path / "empty_with_header.csv"
        empty_with_header.write_text("timeSinceStartup,Node_Head_px\n")
        df = load_continuous_data(empty_with_header)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "timestamp" in df.columns


# ===========================================================================
# load_face_data
# ===========================================================================


class TestLoadFaceData:
    def test_returns_dataframe(self, tmp_face_csv):
        """load_face_data returns a pandas DataFrame."""
        df = load_face_data(tmp_face_csv)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, tmp_face_csv):
        """Result has 'Eyes_Closed_L' and 'Eyes_Closed_R' columns."""
        df = load_face_data(tmp_face_csv)
        assert "Eyes_Closed_L" in df.columns
        assert "Eyes_Closed_R" in df.columns

    def test_missing_file_raises_data_load_error(self, tmp_path):
        """Nonexistent path raises DataLoadError."""
        with pytest.raises(DataLoadError):
            load_face_data(tmp_path / "nonexistent.csv")


# ===========================================================================
# load_session_metadata
# ===========================================================================


class TestLoadSessionMetadata:
    def test_returns_session_metadata(self, tmp_metadata_json):
        """load_session_metadata returns a SessionMetadata instance."""
        meta = load_session_metadata(tmp_metadata_json)
        assert isinstance(meta, SessionMetadata)

    def test_session_id_matches_json(self, tmp_metadata_json):
        """Parsed session_id matches the value in the JSON file."""
        meta = load_session_metadata(tmp_metadata_json)
        assert meta.session_id == "test_session_001"

    def test_feature_flags_parsed(self, tmp_metadata_json):
        """hands_enabled, eyes_enabled, face_enabled are parsed correctly."""
        meta = load_session_metadata(tmp_metadata_json)
        assert meta.hands_enabled is True
        assert meta.eyes_enabled is True
        assert meta.face_enabled is True
        assert meta.body_enabled is False

    def test_missing_file_raises_data_load_error(self, tmp_path):
        """Nonexistent path raises DataLoadError."""
        with pytest.raises(DataLoadError):
            load_session_metadata(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_data_load_error(self, tmp_path):
        """A file containing invalid JSON raises DataLoadError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(DataLoadError):
            load_session_metadata(bad)

    def test_missing_session_id_defaults_to_unknown(self, tmp_path):
        """If 'session_id' key is absent, defaults to 'unknown'."""
        data = {"unity_version": "2022.3.0f1"}  # no session_id
        path = tmp_path / "meta.json"
        with open(path, "w") as f:
            json.dump(data, f)
        meta = load_session_metadata(path)
        assert meta.session_id == "unknown"

    def test_missing_feature_flags_default_to_false(self, tmp_path):
        """Missing boolean feature flags (hands_enabled, etc.) default to False."""
        # Minimal metadata with only session_id — all booleans absent
        data = {"session_id": "s1"}
        path = tmp_path / "meta.json"
        with open(path, "w") as f:
            json.dump(data, f)
        meta = load_session_metadata(path)
        assert meta.hands_enabled is False
        assert meta.eyes_enabled is False
        assert meta.face_enabled is False
        assert meta.body_enabled is False
        assert meta.controllers_enabled is False


# ===========================================================================
# load_events_data
# ===========================================================================


class TestLoadEventsData:
    def test_returns_dataframe(self, tmp_events_csv):
        """load_events_data returns a pandas DataFrame."""
        df = load_events_data(tmp_events_csv)
        assert isinstance(df, pd.DataFrame)

    def test_keeps_name_column(self, tmp_events_csv):
        """'name' is preserved (no rename to trial_type)."""
        df = load_events_data(tmp_events_csv)
        assert "name" in df.columns
        assert "trial_type" not in df.columns

    def test_required_columns_present(self, tmp_events_csv):
        """name, onset, and duration columns are all present."""
        df = load_events_data(tmp_events_csv)
        for col in ("name", "onset", "duration"):
            assert col in df.columns

    def test_onset_sorted_ascending(self, tmp_path):
        """Events are sorted by onset after loading."""
        df_raw = pd.DataFrame(
            {
                "name": ["c", "a", "b"],
                "onset": [10.0, 0.0, 5.0],
                "duration": [1.0, 1.0, 1.0],
            }
        )
        path = tmp_path / "events.csv"
        df_raw.to_csv(path, index=False)
        df = load_events_data(path)
        assert list(df["onset"]) == sorted(df["onset"])

    def test_missing_file_raises_data_load_error(self, tmp_path):
        """Nonexistent path raises DataLoadError."""
        with pytest.raises(DataLoadError):
            load_events_data(tmp_path / "nonexistent.csv")

    def test_missing_required_column_raises(self, tmp_path):
        """CSV missing 'name' column raises DataLoadError."""
        bad = tmp_path / "bad_events.csv"
        pd.DataFrame({"onset": [0.0], "duration": [1.0]}).to_csv(bad, index=False)
        with pytest.raises(DataLoadError, match="missing"):
            load_events_data(bad)

    def test_non_numeric_onset_raises(self, tmp_path):
        """Non-numeric 'onset' column raises DataLoadError."""
        bad = tmp_path / "bad_events2.csv"
        pd.DataFrame(
            {
                "name": ["evt"],
                "onset": ["not_a_number"],
                "duration": [1.0],
            }
        ).to_csv(bad, index=False)
        with pytest.raises(DataLoadError):
            load_events_data(bad)


# ===========================================================================
# load_session
# ===========================================================================


class TestLoadSession:
    def _input_config(self, tmp_path: Path) -> InputConfig:
        return InputConfig(
            data_dir=tmp_path,
            continuous_data_pattern="*_ContinuousData.csv",
            face_data_pattern="*_FaceExpressionData.csv",
            metadata_pattern="session_metadata.json",
            events_data_pattern="*_EventsData.csv",
        )

    def test_returns_session(self, tmp_session_dir, tmp_path):
        """load_session returns a Session object."""
        config = self._input_config(tmp_path)
        session = load_session(tmp_session_dir, config)
        assert isinstance(session, Session)

    def test_session_id_from_metadata(self, tmp_session_dir, tmp_path):
        """The session_id matches the value in session_metadata.json."""
        config = self._input_config(tmp_path)
        session = load_session(tmp_session_dir, config)
        assert session.session_id == "test_session_001"

    def test_raw_continuous_data_loaded(self, tmp_session_dir, tmp_path):
        """raw_continuous_data is populated and has a 'timestamp' column."""
        config = self._input_config(tmp_path)
        session = load_session(tmp_session_dir, config)
        assert session.raw_continuous_data is not None
        assert "timestamp" in session.raw_continuous_data.columns

    def test_raw_face_data_loaded(self, tmp_session_dir, tmp_path):
        """raw_face_data is populated when a face CSV is present."""
        config = self._input_config(tmp_path)
        session = load_session(tmp_session_dir, config)
        assert session.raw_face_data is not None

    def test_raw_events_data_loaded(self, tmp_session_dir, tmp_path):
        """raw_events_data is populated when an events CSV is present."""
        config = self._input_config(tmp_path)
        session = load_session(tmp_session_dir, config)
        assert session.raw_events_data is not None

    def test_missing_directory_raises(self, tmp_path):
        """Passing a nonexistent directory raises MissingDataError."""
        config = self._input_config(tmp_path)
        with pytest.raises(MissingDataError):
            load_session(tmp_path / "does_not_exist", config)

    def test_missing_metadata_file_raises(self, tmp_path):
        """Session directory without metadata file raises MissingDataError."""
        # Create a directory with only continuous data — no metadata
        session_dir = tmp_path / "session_no_meta"
        session_dir.mkdir()
        pd.DataFrame({"timeSinceStartup": [1.0], "Node_Head_px": [0.1]}).to_csv(
            session_dir / "test_ContinuousData.csv", index=False
        )
        config = self._input_config(tmp_path)
        with pytest.raises(MissingDataError, match="metadata"):
            load_session(session_dir, config)

    def test_missing_continuous_file_raises(self, tmp_path, tmp_metadata_json):
        """Session directory without continuous data raises MissingDataError."""
        session_dir = tmp_path / "session_no_continuous"
        session_dir.mkdir()
        import shutil

        shutil.copy(tmp_metadata_json, session_dir / "session_metadata.json")
        config = self._input_config(tmp_path)
        with pytest.raises(MissingDataError, match="continuous"):
            load_session(session_dir, config)


# ===========================================================================
# load_custom_tables_json
# ===========================================================================

from resxr.io.readers import (  # noqa: E402
    find_custom_class_csvs,
    load_custom_class_csv,
    load_custom_tables_json,
)

_FIXTURES = Path(__file__).parent / "_fixtures"


class TestLoadCustomTablesJson:
    def test_valid_file_parses_all_fields(self):
        tables = load_custom_tables_json(_FIXTURES / "sample_custom_tables.json")
        assert tables is not None
        assert len(tables) == 1
        t = tables[0]
        assert t.class_name == "ChoiceEvent"
        assert t.row_count == 2
        rt = t.columns[0]
        assert rt.name == "reaction_time"
        assert rt.units == "s"
        assert rt.minimum == 0.0
        assert rt.maximum is None
        assert rt.levels is None
        choice = t.columns[1]
        assert choice.levels == {"L": "left", "R": "right"}
        assert choice.units is None

    def test_absent_file_returns_none(self, tmp_path):
        assert load_custom_tables_json(tmp_path / "nope.json") is None

    def test_malformed_json_returns_none(self, tmp_path):
        bad = tmp_path / "custom_tables.json"
        bad.write_text("{not valid")
        assert load_custom_tables_json(bad) is None

    def test_missing_required_key_returns_none(self, tmp_path):
        bad = tmp_path / "custom_tables.json"
        bad.write_text('[{"row_count": 1, "columns": []}]')  # no class_name
        assert load_custom_tables_json(bad) is None


# ===========================================================================
# find_custom_class_csvs
# ===========================================================================


class TestFindCustomClassCsvs:
    def _make(self, d: Path, name: str, rows: int = 2):
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"onset": [0.0, 1.0][:rows], "duration": [0.0, 0.0][:rows]})
        df.to_csv(d / name, index=False)

    def test_loads_all_csvs_and_strips_prefix(self, tmp_path):
        custom = tmp_path / "custom_tables"
        self._make(custom, "2026.03.12_15-47_ChoiceEvent.csv")
        self._make(custom, "2026.03.12_15-47_TrialsData.csv")
        result = find_custom_class_csvs(custom, recording_id="2026.03.12_15-47")
        assert set(result.keys()) == {"ChoiceEvent", "TrialsData"}

    def test_bare_filename_no_prefix(self, tmp_path):
        custom = tmp_path / "custom_tables"
        self._make(custom, "ChoiceEvent.csv")
        result = find_custom_class_csvs(custom, recording_id="anything")
        assert set(result.keys()) == {"ChoiceEvent"}

    def test_missing_folder_returns_empty(self, tmp_path):
        assert find_custom_class_csvs(tmp_path / "nope", recording_id="x") == {}

    def test_ignores_non_csv_files(self, tmp_path):
        custom = tmp_path / "custom_tables"
        self._make(custom, "ChoiceEvent.csv")
        (custom / "custom_tables.json").write_text("[]")
        (custom / "notes.txt").write_text("x")
        result = find_custom_class_csvs(custom, recording_id="x")
        assert set(result.keys()) == {"ChoiceEvent"}


# ===========================================================================
# load_custom_class_csv
# ===========================================================================


class TestLoadCustomClassCsv:
    def test_loads_valid(self, tmp_path):
        p = tmp_path / "ChoiceEvent.csv"
        pd.DataFrame({"onset": [0.0], "duration": [1.0], "rt": [0.3]}).to_csv(p, index=False)
        df = load_custom_class_csv(p)
        assert list(df["onset"]) == [0.0]

    def test_missing_onset_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"duration": [1.0]}).to_csv(p, index=False)
        with pytest.raises(DataLoadError, match="onset"):
            load_custom_class_csv(p)

    def test_missing_duration_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"onset": [0.0]}).to_csv(p, index=False)
        with pytest.raises(DataLoadError, match="duration"):
            load_custom_class_csv(p)

    def test_non_numeric_onset_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"onset": ["x"], "duration": [1.0]}).to_csv(p, index=False)
        with pytest.raises(DataLoadError):
            load_custom_class_csv(p)


# ===========================================================================
# discover_sessions
# ===========================================================================


class TestDiscoverSessions:
    def _input_config(self, data_dir: Path) -> InputConfig:
        return InputConfig(
            data_dir=data_dir,
            continuous_data_pattern="*_ContinuousData.csv",
            face_data_pattern="*_FaceExpressionData.csv",
            metadata_pattern="session_metadata.json",
            events_data_pattern="*_EventsData.csv",
        )

    def test_finds_all_session_directories(self, tmp_path):
        """Returns one path per subdirectory that contains a metadata file."""
        data_dir = tmp_path / "sessions"
        data_dir.mkdir()
        for i in range(3):
            d = data_dir / f"session_{i:03d}"
            d.mkdir()
            (d / "session_metadata.json").write_text(json.dumps({"session_id": f"s{i}"}))
        sessions = discover_sessions(self._input_config(data_dir))
        assert len(sessions) == 3

    def test_empty_data_dir_returns_empty_list(self, tmp_path):
        """A directory with no subdirs returns an empty list."""
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        sessions = discover_sessions(self._input_config(data_dir))
        assert sessions == []

    def test_ignores_dirs_without_metadata(self, tmp_path):
        """Subdirectories without the metadata file are excluded."""
        data_dir = tmp_path / "sessions"
        data_dir.mkdir()
        # One valid session, one without metadata
        good = data_dir / "good_session"
        good.mkdir()
        (good / "session_metadata.json").write_text(json.dumps({"session_id": "s1"}))
        bad = data_dir / "no_metadata"
        bad.mkdir()
        (bad / "random_file.txt").write_text("nope")

        sessions = discover_sessions(self._input_config(data_dir))
        assert len(sessions) == 1
        assert sessions[0] == good

    def test_nonexistent_data_dir_raises(self, tmp_path):
        """Passing a nonexistent data_dir raises MissingDataError."""
        with pytest.raises(MissingDataError):
            discover_sessions(self._input_config(tmp_path / "does_not_exist"))

    def test_returns_list_of_paths(self, tmp_path):
        """Return value is a list of Path objects."""
        data_dir = tmp_path / "sessions"
        data_dir.mkdir()
        d = data_dir / "session_001"
        d.mkdir()
        (d / "session_metadata.json").write_text(json.dumps({"session_id": "s1"}))
        sessions = discover_sessions(self._input_config(data_dir))
        assert all(isinstance(p, Path) for p in sessions)


# ===========================================================================
# load_session — custom-table consistency logging (no validation check)
# ===========================================================================


class TestLoadSessionCustomTableLogging:
    def _config(self, data_dir: Path) -> InputConfig:
        return InputConfig(
            data_dir=data_dir,
            continuous_data_pattern="*_ContinuousData.csv",
            face_data_pattern="*_FaceExpressionData.csv",
            metadata_pattern="session_metadata.json",
            events_data_pattern="*_EventsData.csv",
            custom_tables_dir="custom_tables",
        )

    def _session_dir(self, tmp_path: Path, choice_rows: int, tables_json: str) -> Path:
        d = tmp_path / "sess"
        d.mkdir()
        (d / "session_metadata.json").write_text(json.dumps({"session_id": "sess"}))
        pd.DataFrame({"timeSinceStartup": [1.0, 1.1], "Node_Head_px": [0.1, 0.2]}).to_csv(
            d / "sess_ContinuousData.csv", index=False
        )
        custom = d / "custom_tables"
        custom.mkdir()
        pd.DataFrame(
            {"onset": [0.0, 1.0][:choice_rows], "duration": [0.0, 0.0][:choice_rows]}
        ).to_csv(custom / "sess_ChoiceEvent.csv", index=False)
        (custom / "custom_tables.json").write_text(tables_json)
        return d

    def test_row_count_mismatch_logs_warning(self, tmp_path, caplog):
        import logging

        d = self._session_dir(
            tmp_path,
            choice_rows=1,
            tables_json='[{"class_name":"ChoiceEvent","row_count":5,"columns":[]}]',
        )
        with caplog.at_level(logging.WARNING):
            load_session(d, self._config(tmp_path))
        assert "row_count" in caplog.text

    def test_declared_class_without_csv_logs_warning(self, tmp_path, caplog):
        import logging

        d = self._session_dir(
            tmp_path,
            choice_rows=1,
            tables_json=(
                '[{"class_name":"ChoiceEvent","row_count":1,"columns":[]},'
                '{"class_name":"Ghost","row_count":2,"columns":[]}]'
            ),
        )
        with caplog.at_level(logging.WARNING):
            load_session(d, self._config(tmp_path))
        assert "Ghost" in caplog.text
        assert "no matching CSV" in caplog.text

    def test_consistent_schema_no_warning(self, tmp_path, caplog):
        import logging

        d = self._session_dir(
            tmp_path,
            choice_rows=1,
            tables_json='[{"class_name":"ChoiceEvent","row_count":1,"columns":[]}]',
        )
        with caplog.at_level(logging.WARNING):
            load_session(d, self._config(tmp_path))
        assert "row_count" not in caplog.text
        assert "no matching CSV" not in caplog.text
