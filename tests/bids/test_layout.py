"""Tests for BIDSLayout directory management (bids/layout.py)."""

from __future__ import annotations

from pathlib import Path

from resxr.bids.layout import BIDSLayout
from resxr.core.constants import TrackingSystem

# ===========================================================================
# BIDSLayout path computation
# ===========================================================================


class TestBIDSLayoutPaths:
    def test_get_subject_dir(self, tmp_path, minimal_session):
        """get_subject_dir ends in 'sub-{subject_id}'."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_subject_dir(minimal_session)
        assert path.name == f"sub-{minimal_session.subject_id}"

    def test_get_subject_dir_under_bids_root(self, tmp_path, minimal_session):
        """get_subject_dir is directly under bids_root (raw)."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_subject_dir(minimal_session)
        assert path.parent == tmp_path

    def test_get_subject_dir_derivative(self, tmp_path, minimal_session):
        """Derivative subject dir is under derivatives/resxr."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_subject_dir(minimal_session, derivative=True)
        assert "derivatives" in str(path)
        assert "resxr" in str(path)

    def test_get_session_dir_contains_ses(self, tmp_path, minimal_session):
        """get_session_dir path ends in 'ses-{session_label}'."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_session_dir(minimal_session)
        assert path.name == f"ses-{minimal_session.session_label}"

    def test_get_session_dir_under_subject_dir(self, tmp_path, minimal_session):
        """get_session_dir is directly under the subject directory."""
        layout = BIDSLayout(tmp_path, "vr")
        session_dir = layout.get_session_dir(minimal_session)
        subject_dir = layout.get_subject_dir(minimal_session)
        assert session_dir.parent == subject_dir

    def test_get_motion_dir_ends_in_motion(self, tmp_path, minimal_session):
        """get_motion_dir path ends in 'motion'."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_motion_dir(minimal_session)
        assert path.name == "motion"

    def test_get_motion_dir_under_session_dir(self, tmp_path, minimal_session):
        """get_motion_dir is directly under the session directory."""
        layout = BIDSLayout(tmp_path, "vr")
        motion_dir = layout.get_motion_dir(minimal_session)
        session_dir = layout.get_session_dir(minimal_session)
        assert motion_dir.parent == session_dir

    def test_all_path_methods_return_path_objects(self, tmp_path, minimal_session):
        """All path-returning methods return pathlib.Path objects."""
        layout = BIDSLayout(tmp_path, "vr")
        assert isinstance(layout.get_subject_dir(minimal_session), Path)
        assert isinstance(layout.get_session_dir(minimal_session), Path)
        assert isinstance(layout.get_motion_dir(minimal_session), Path)

    def test_custom_root_respected(self, tmp_path, minimal_session):
        """All returned paths are under the provided bids_root."""
        layout = BIDSLayout(tmp_path / "custom_root", "vr")
        assert str(layout.get_subject_dir(minimal_session)).startswith(
            str(tmp_path / "custom_root")
        )

    def test_get_motion_file_path(self, tmp_path, minimal_session):
        """get_motion_file returns a path with the correct filename format."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_motion_file(minimal_session, TrackingSystem.HEAD)
        assert path.name == "sub-01_ses-01_task-vr_tracksys-Head_motion.tsv"

    def test_get_channels_file_path(self, tmp_path, minimal_session):
        """get_channels_file returns a path with the correct filename format."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_channels_file(minimal_session, TrackingSystem.HEAD)
        assert path.name == "sub-01_ses-01_task-vr_tracksys-Head_channels.tsv"

    def test_get_scans_file(self, tmp_path, minimal_session):
        """get_scans_file returns a path ending in _scans.tsv."""
        layout = BIDSLayout(tmp_path, "vr")
        path = layout.get_scans_file(minimal_session)
        assert path.name.endswith("_scans.tsv")
        assert "sub-01" in path.name


# ===========================================================================
# BIDSLayout.create_structure
# ===========================================================================


class TestCreateStructure:
    def test_creates_motion_directory(self, tmp_path, minimal_session):
        """create_structure creates the motion directory on disk."""
        layout = BIDSLayout(tmp_path, "vr")
        layout.create_structure(minimal_session)
        assert layout.get_motion_dir(minimal_session).is_dir()

    def test_creates_subject_and_session_dirs(self, tmp_path, minimal_session):
        """create_structure also creates the subject and session directories."""
        layout = BIDSLayout(tmp_path, "vr")
        layout.create_structure(minimal_session)
        assert layout.get_subject_dir(minimal_session).is_dir()
        assert layout.get_session_dir(minimal_session).is_dir()

    def test_idempotent(self, tmp_path, minimal_session):
        """Calling create_structure twice does not raise an error."""
        layout = BIDSLayout(tmp_path, "vr")
        layout.create_structure(minimal_session)
        layout.create_structure(minimal_session)  # Should not raise
        assert layout.get_motion_dir(minimal_session).is_dir()

    def test_creates_derivative_structure(self, tmp_path, minimal_session):
        """derivative=True creates directories under derivatives/resxr/."""
        layout = BIDSLayout(tmp_path, "vr")
        layout.create_structure(minimal_session, derivative=True)
        deriv_motion = layout.get_motion_dir(minimal_session, derivative=True)
        assert deriv_motion.is_dir()
        assert "derivatives" in str(deriv_motion)


def test_sourcedata_session_dir(minimal_session, tmp_path):
    from resxr.bids.layout import BIDSLayout

    layout = BIDSLayout(tmp_path, "vr")
    p = layout.sourcedata_session_dir(minimal_session)
    assert p == tmp_path / "sourcedata" / "sub-01" / "ses-01"
