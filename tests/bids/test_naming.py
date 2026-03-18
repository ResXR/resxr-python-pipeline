"""Tests for BIDS filename generators (bids/naming.py)."""

from __future__ import annotations

import pytest

from resxr.bids.naming import make_bids_filename, make_channels_filename, make_motion_filename
from resxr.core.constants import TrackingSystem

# ===========================================================================
# make_bids_filename (the low-level generator)
# ===========================================================================


@pytest.mark.parametrize(
    "subject,session,task,system,suffix,ext,expected",
    [
        (
            "01",
            "01",
            "vr",
            TrackingSystem.HEAD,
            "motion",
            "tsv",
            "sub-01_ses-01_task-vr_tracksys-Head_motion.tsv",
        ),
        (
            "02",
            "baseline",
            "rest",
            TrackingSystem.HANDS,
            "channels",
            "tsv",
            "sub-02_ses-baseline_task-rest_tracksys-Hands_channels.tsv",
        ),
        (
            "01",
            "01",
            "VRtracking",
            TrackingSystem.EYES,
            "motion",
            "json",
            "sub-01_ses-01_task-VRtracking_tracksys-Eyes_motion.json",
        ),
    ],
)
def test_make_bids_filename_format(subject, session, task, system, suffix, ext, expected):
    """make_bids_filename produces exactly the expected string."""
    result = make_bids_filename(subject, session, task, system, suffix, ext)
    assert result == expected


def test_make_bids_filename_no_spaces():
    """Filename contains no spaces."""
    result = make_bids_filename("01", "01", "vr", TrackingSystem.HEAD, "motion", "tsv")
    assert " " not in result


def test_make_bids_filename_has_all_bids_parts():
    """Filename contains sub-, ses-, task-, and tracksys- parts."""
    result = make_bids_filename("01", "01", "vr", TrackingSystem.HEAD, "motion", "tsv")
    assert result.startswith("sub-01_")
    assert "ses-01_" in result
    assert "task-vr_" in result
    assert "tracksys-Head_" in result


# ===========================================================================
# make_motion_filename
# ===========================================================================


@pytest.mark.parametrize(
    "subject,session,task,system,expected",
    [
        ("01", "01", "vr", TrackingSystem.HEAD, "sub-01_ses-01_task-vr_tracksys-Head_motion.tsv"),
        ("03", "pre", "xr", TrackingSystem.FACE, "sub-03_ses-pre_task-xr_tracksys-Face_motion.tsv"),
    ],
)
def test_make_motion_filename_format(subject, session, task, system, expected):
    """make_motion_filename produces the expected filename."""
    assert make_motion_filename(subject, session, task, system) == expected


def test_make_motion_filename_default_extension_is_tsv():
    """Default extension for motion file is 'tsv'."""
    name = make_motion_filename("01", "01", "vr", TrackingSystem.HEAD)
    assert name.endswith(".tsv")


def test_make_motion_filename_json_extension():
    """Passing extension='json' gives a .json file."""
    name = make_motion_filename("01", "01", "vr", TrackingSystem.HEAD, extension="json")
    assert name.endswith(".json")


def test_make_motion_filename_ends_with_motion():
    """Filename contains '_motion.' before the extension."""
    name = make_motion_filename("01", "01", "vr", TrackingSystem.HEAD)
    assert "_motion." in name


# ===========================================================================
# make_channels_filename
# ===========================================================================


@pytest.mark.parametrize(
    "subject,session,task,system,expected",
    [
        ("01", "01", "vr", TrackingSystem.HEAD, "sub-01_ses-01_task-vr_tracksys-Head_channels.tsv"),
        (
            "02",
            "01",
            "vr",
            TrackingSystem.HANDS,
            "sub-02_ses-01_task-vr_tracksys-Hands_channels.tsv",
        ),
    ],
)
def test_make_channels_filename_format(subject, session, task, system, expected):
    """make_channels_filename produces the expected filename."""
    assert make_channels_filename(subject, session, task, system) == expected


def test_make_channels_filename_default_extension_is_tsv():
    """Default extension for channels file is 'tsv'."""
    name = make_channels_filename("01", "01", "vr", TrackingSystem.HEAD)
    assert name.endswith(".tsv")


def test_make_channels_filename_ends_with_channels():
    """Filename contains '_channels.' before the extension."""
    name = make_channels_filename("01", "01", "vr", TrackingSystem.HEAD)
    assert "_channels." in name


# ===========================================================================
# Consistency between motion and channels filenames
# ===========================================================================


def test_motion_and_channels_share_prefix():
    """motion.tsv and channels.tsv for the same inputs share the same BIDS prefix."""
    motion = make_motion_filename("01", "01", "vr", TrackingSystem.HEAD)
    channels = make_channels_filename("01", "01", "vr", TrackingSystem.HEAD)
    # Both should start with sub-01_ses-01_task-vr_tracksys-Head_
    prefix = "sub-01_ses-01_task-vr_tracksys-Head_"
    assert motion.startswith(prefix)
    assert channels.startswith(prefix)


@pytest.mark.parametrize("system", list(TrackingSystem))
def test_all_systems_produce_valid_filenames(system):
    """Every TrackingSystem member produces a non-empty filename."""
    name = make_motion_filename("01", "01", "vr", system)
    assert isinstance(name, str)
    assert len(name) > 0
    assert name.endswith(".tsv")


# ===========================================================================
# Special characters in subject IDs
# ===========================================================================


def test_special_characters_in_subject_id_pass_through():
    """make_bids_filename does not sanitise subject_id — characters pass through as-is.

    NOTE: BIDS spec restricts entity values to alphanumeric characters.
    This test documents current behaviour (no validation), not desired behaviour.
    A future input-sanitisation layer should enforce BIDS-legal characters.
    """
    name = make_bids_filename("01+special", "01", "vr", TrackingSystem.HEAD, "motion", "tsv")
    assert "sub-01+special_" in name


@pytest.mark.parametrize("subject_id", ["01", "AB", "sub01", "P001"])
def test_alphanumeric_subject_ids_produce_valid_filenames(subject_id):
    """Standard alphanumeric subject IDs produce well-formed BIDS filenames."""
    name = make_motion_filename(subject_id, "01", "vr", TrackingSystem.HEAD)
    assert name.startswith(f"sub-{subject_id}_")
    assert name.endswith(".tsv")
