"""Tests for BIDS file writer functions (io/writers.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from resxr.io.writers import (
    write_bids_events,
    write_bids_tsv,
    write_channels_tsv,
    write_json,
    write_motion_tsv,
    write_participants_tsv,
    write_scans_tsv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col_a": [1.0, 2.0, 3.0],
            "col_b": [4.0, 5.0, 6.0],
        }
    )


def _events_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_type": ["start", "end"],
            "onset": [0.0, 10.0],
            "duration": [0.0, 0.0],
        }
    )


# ===========================================================================
# write_bids_tsv
# ===========================================================================


class TestWriteBidsTsv:
    def test_creates_file(self, tmp_path):
        """write_bids_tsv creates the output file."""
        path = tmp_path / "out.tsv"
        write_bids_tsv(_sample_df(), path)
        assert path.exists()

    def test_tab_separated(self, tmp_path):
        """Written file uses tabs as the column separator."""
        path = tmp_path / "out.tsv"
        write_bids_tsv(_sample_df(), path)
        content = path.read_text()
        first_line = content.splitlines()[0]
        assert "\t" in first_line

    def test_no_row_index(self, tmp_path):
        """Written file does not include a row index column."""
        path = tmp_path / "out.tsv"
        write_bids_tsv(_sample_df(), path)
        df_read = pd.read_csv(path, sep="\t")
        # Row index would appear as an 'Unnamed: 0' column
        assert "Unnamed: 0" not in df_read.columns

    def test_roundtrip(self, tmp_path):
        """Data read back after writing matches the original DataFrame."""
        df = _sample_df()
        path = tmp_path / "out.tsv"
        write_bids_tsv(df, path)
        df_read = pd.read_csv(path, sep="\t")
        pd.testing.assert_frame_equal(df.reset_index(drop=True), df_read)

    def test_with_header(self, tmp_path):
        """include_header=True writes column names on the first line."""
        path = tmp_path / "out.tsv"
        write_bids_tsv(_sample_df(), path, include_header=True)
        first_line = path.read_text().splitlines()[0]
        assert "col_a" in first_line

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories are created automatically."""
        path = tmp_path / "sub" / "ses" / "motion" / "out.tsv"
        write_bids_tsv(_sample_df(), path)
        assert path.exists()


# ===========================================================================
# write_json
# ===========================================================================


class TestWriteJson:
    def test_creates_file(self, tmp_path):
        """write_json creates the output file."""
        path = tmp_path / "out.json"
        write_json({"key": "value"}, path)
        assert path.exists()

    def test_valid_json(self, tmp_path):
        """Written file contains valid JSON."""
        path = tmp_path / "out.json"
        write_json({"TaskName": "vr", "SamplingFrequency": 90.0}, path)
        data = json.loads(path.read_text())
        assert data["TaskName"] == "vr"

    def test_nested_dict_preserved(self, tmp_path):
        """Nested dicts are round-tripped correctly."""
        nested = {"outer": {"inner": [1, 2, 3]}}
        path = tmp_path / "nested.json"
        write_json(nested, path)
        assert json.loads(path.read_text()) == nested

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories are created automatically."""
        path = tmp_path / "deep" / "nested" / "out.json"
        write_json({"x": 1}, path)
        assert path.exists()


# ===========================================================================
# write_motion_tsv
# ===========================================================================


class TestWriteMotionTsv:
    def test_creates_file(self, tmp_path):
        """write_motion_tsv creates the output file."""
        path = tmp_path / "motion.tsv"
        write_motion_tsv(_sample_df(), path, missing_values="n/a")
        assert path.exists()

    def test_no_header_row(self, tmp_path):
        """Motion TSV has no header: first line contains only data values."""
        df = pd.DataFrame({"latency": [0.0, 0.1], "Node_Head_px": [0.1, 0.2]})
        path = tmp_path / "motion.tsv"
        write_motion_tsv(df, path, missing_values="n/a")
        first_line = path.read_text().splitlines()[0]
        # If no header, the first line should NOT contain column names
        assert "latency" not in first_line
        assert "Node_Head_px" not in first_line

    def test_tab_separated(self, tmp_path):
        """Motion TSV uses tabs as delimiters."""
        path = tmp_path / "motion.tsv"
        write_motion_tsv(_sample_df(), path, missing_values="n/a")
        assert "\t" in path.read_text().splitlines()[0]

    def test_nan_written_as_missing_values(self, tmp_path):
        """NaN values are written using the configured missing_values string."""
        df = pd.DataFrame({"latency": [0.0, float("nan")], "val": [1.0, 2.0]})
        path = tmp_path / "motion.tsv"
        write_motion_tsv(df, path, missing_values="n/a")
        content = path.read_text()
        assert "n/a" in content


# ===========================================================================
# write_channels_tsv
# ===========================================================================


class TestWriteChannelsTsv:
    def test_creates_file(self, tmp_path):
        """write_channels_tsv creates the output file."""
        path = tmp_path / "channels.tsv"
        df = pd.DataFrame(
            {
                "name": ["latency", "Node_Head_px"],
                "type": ["LATENCY", "POS"],
                "units": ["s", "m"],
            }
        )
        write_channels_tsv(df, path)
        assert path.exists()

    def test_has_header(self, tmp_path):
        """Channels TSV includes column headers."""
        path = tmp_path / "channels.tsv"
        df = pd.DataFrame({"name": ["latency"], "type": ["LATENCY"], "units": ["s"]})
        write_channels_tsv(df, path)
        first_line = path.read_text().splitlines()[0]
        assert "name" in first_line


# ===========================================================================
# write_scans_tsv
# ===========================================================================


class TestWriteScansTsv:
    def test_creates_file(self, tmp_path):
        """write_scans_tsv creates the output file."""
        path = tmp_path / "scans.tsv"
        df = pd.DataFrame(
            {
                "filename": ["motion/sub-01_ses-01_motion.tsv"],
                "acq_time": ["n/a"],
            }
        )
        write_scans_tsv(df, path)
        assert path.exists()

    def test_contains_filename_column(self, tmp_path):
        """Scans TSV includes the 'filename' column."""
        path = tmp_path / "scans.tsv"
        df = pd.DataFrame(
            {
                "filename": ["motion/file1.tsv", "motion/file2.tsv"],
                "acq_time": ["n/a", "n/a"],
            }
        )
        write_scans_tsv(df, path)
        df_read = pd.read_csv(path, sep="\t")
        assert "filename" in df_read.columns
        assert len(df_read) == 2


# ===========================================================================
# write_participants_tsv
# ===========================================================================


class TestWriteParticipantsTsv:
    def test_creates_file(self, tmp_path):
        """write_participants_tsv creates the output file."""
        path = tmp_path / "participants.tsv"
        df = pd.DataFrame(
            {
                "participant_id": ["sub-01"],
                "age": ["25"],
                "sex": ["M"],
            }
        )
        write_participants_tsv(df, path)
        assert path.exists()

    def test_participant_id_column_present(self, tmp_path):
        """Written file has the participant_id column."""
        path = tmp_path / "participants.tsv"
        df = pd.DataFrame({"participant_id": ["sub-01", "sub-02"]})
        write_participants_tsv(df, path)
        df_read = pd.read_csv(path, sep="\t")
        assert "participant_id" in df_read.columns


# ===========================================================================
# write_bids_events
# ===========================================================================


class TestWriteBidsEvents:
    def test_creates_tsv_file(self, tmp_path):
        """write_bids_events creates the events.tsv file."""
        path = tmp_path / "events.tsv"
        write_bids_events(_events_df(), path)
        assert path.exists()

    def test_creates_json_sidecar(self, tmp_path):
        """write_bids_events also creates a _events.json sidecar."""
        path = tmp_path / "events.tsv"
        write_bids_events(_events_df(), path)
        assert (tmp_path / "events.json").exists()

    def test_required_columns_present(self, tmp_path):
        """Written TSV has trial_type, onset, and duration columns."""
        path = tmp_path / "events.tsv"
        write_bids_events(_events_df(), path)
        df = pd.read_csv(path, sep="\t")
        for col in ("trial_type", "onset", "duration"):
            assert col in df.columns

    def test_missing_required_column_raises(self, tmp_path):
        """DataFrame missing 'onset' column raises BIDSWriteError."""
        from resxr.core.exceptions import BIDSWriteError

        bad_df = pd.DataFrame({"trial_type": ["evt"], "duration": [1.0]})
        with pytest.raises(BIDSWriteError):
            write_bids_events(bad_df, tmp_path / "events.tsv")

    def test_events_sorted_by_onset(self, tmp_path):
        """Events in the output file are sorted by onset time."""
        df = pd.DataFrame(
            {
                "trial_type": ["c", "a", "b"],
                "onset": [10.0, 0.0, 5.0],
                "duration": [0.0, 0.0, 0.0],
            }
        )
        path = tmp_path / "events.tsv"
        write_bids_events(df, path)
        df_read = pd.read_csv(path, sep="\t")
        assert list(df_read["onset"]) == sorted(df_read["onset"])
