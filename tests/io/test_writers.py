"""Tests for BIDS file writer functions (io/writers.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from resxr.io.writers import (
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
            "name": ["start", "end"],
            "onset": [0.0, 10.0],
            "duration": [0.0, 0.0],
        }
    )


# ===========================================================================
# Line endings (cross-platform reproducibility)
# ===========================================================================


class TestLineEndingsAreLF:
    """
    All BIDS text outputs must use LF line endings on every OS so the dataset is
    byte-reproducible across Windows and Linux.

    Note: on Linux this passes even without the explicit ``lineterminator`` /
    ``newline`` arguments (text mode already emits LF); the value of this test is
    as a cross-platform regression guard, since ``pandas.to_csv`` otherwise
    defaults to ``os.linesep`` (CRLF on Windows).
    """

    def test_motion_tsv_is_lf(self, tmp_path):
        path = tmp_path / "x_motion.tsv"
        write_motion_tsv(_sample_df(), path, missing_values="n/a")
        data = path.read_bytes()
        assert b"\r\n" not in data
        assert b"\n" in data

    def test_channels_tsv_is_lf(self, tmp_path):
        path = tmp_path / "x_channels.tsv"
        write_channels_tsv(_sample_df(), path)
        assert b"\r\n" not in path.read_bytes()

    def test_json_is_lf(self, tmp_path):
        path = tmp_path / "x.json"
        write_json({"a": 1, "b": {"c": 2}}, path)
        assert b"\r\n" not in path.read_bytes()


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
    def _wide(self):
        return pd.DataFrame(
            {
                "onset": [0.0, 5.0],
                "duration": [0.0, 1.0],
                "name": ["start", "ChoiceEvent"],
                "reaction_time": ["n/a", 0.3],
            }
        )

    def _sidecar(self):
        return {"onset": {"Units": "s"}, "name": {"Description": "Event"}}

    def test_writes_tsv_and_json(self, tmp_path):
        from resxr.io.writers import write_bids_events

        path = tmp_path / "events.tsv"
        write_bids_events(self._wide(), path, self._sidecar())
        assert path.exists()
        assert (tmp_path / "events.json").exists()

    def test_extra_columns_not_dropped(self, tmp_path):
        from resxr.io.writers import write_bids_events

        path = tmp_path / "events.tsv"
        write_bids_events(self._wide(), path, self._sidecar())
        df = pd.read_csv(path, sep="\t")
        assert "reaction_time" in df.columns
        assert "name" in df.columns
        assert "trial_type" not in df.columns

    def test_sidecar_content_matches(self, tmp_path):
        import json as _json

        from resxr.io.writers import write_bids_events

        path = tmp_path / "events.tsv"
        write_bids_events(self._wide(), path, self._sidecar())
        assert _json.loads((tmp_path / "events.json").read_text()) == self._sidecar()

    def test_missing_required_column_raises(self, tmp_path):
        from resxr.core.exceptions import BIDSWriteError
        from resxr.io.writers import write_bids_events

        bad = pd.DataFrame({"name": ["x"], "duration": [1.0]})
        with pytest.raises(BIDSWriteError):
            write_bids_events(bad, tmp_path / "events.tsv", {})

    def test_non_tsv_path_raises_valueerror(self, tmp_path):
        from resxr.io.writers import write_bids_events

        with pytest.raises(ValueError):
            write_bids_events(self._wide(), tmp_path / "events.csv", {})


# ===========================================================================
# copy_sourcedata
# ===========================================================================


class TestCopySourcedata:
    def test_copies_verbatim_including_subdirs(self, tmp_path):
        from resxr.io.writers import copy_sourcedata

        src = tmp_path / "src"
        (src / "sub").mkdir(parents=True)
        (src / "a.csv").write_text("x,y\n1,2\n")
        (src / "sub" / "b.json").write_text('{"k": 1}')
        dest = tmp_path / "dest"
        copy_sourcedata(src, dest)
        assert (dest / "a.csv").read_text() == "x,y\n1,2\n"
        assert (dest / "sub" / "b.json").read_text() == '{"k": 1}'

    def test_skips_when_dest_nonempty_and_no_overwrite(self, tmp_path, caplog):
        import logging

        from resxr.io.writers import copy_sourcedata

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.csv").write_text("new")
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "existing.csv").write_text("old")
        with caplog.at_level(logging.WARNING):
            copy_sourcedata(src, dest, overwrite=False)
        assert (dest / "existing.csv").exists()
        assert not (dest / "a.csv").exists()
        assert "skipping" in caplog.text.lower()

    def test_recopies_when_overwrite_true(self, tmp_path):
        from resxr.io.writers import copy_sourcedata

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.csv").write_text("new")
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "existing.csv").write_text("old")
        copy_sourcedata(src, dest, overwrite=True)
        assert (dest / "a.csv").exists()
