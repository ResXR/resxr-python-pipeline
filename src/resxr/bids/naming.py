"""
BIDS filename conventions for ResXR pipeline.

Generates BIDS-compliant filenames for motion data files.
"""

from __future__ import annotations

from ..core.constants import TrackingSystem


def make_bids_filename(
    subject_id: str,
    session_label: str,
    task_name: str,
    system: TrackingSystem,
    suffix: str,
    extension: str,
) -> str:
    """
    Generate a BIDS-compliant filename.

    Parameters
    ----------
    subject_id : str
        Subject identifier (without 'sub-' prefix)
    session_label : str
        Session label (without 'ses-' prefix)
    task_name : str
        Task name (e.g., "VRtracking")
    system : TrackingSystem
        Tracking system
    suffix : str
        File suffix (e.g., "motion", "channels")
    extension : str
        File extension (e.g., "tsv", "json")

    Returns
    -------
    str
        BIDS-compliant filename
    """
    return (
        f"sub-{subject_id}_"
        f"ses-{session_label}_"
        f"task-{task_name}_"
        f"tracksys-{system.value}_"
        f"{suffix}.{extension}"
    )


def make_motion_filename(
    subject_id: str,
    session_label: str,
    task_name: str,
    system: TrackingSystem,
    extension: str = "tsv",
) -> str:
    """
    Generate BIDS motion data filename.

    Example: sub-01_ses-01_task-VRtracking_tracksys-Head_motion.tsv
    """
    return make_bids_filename(subject_id, session_label, task_name, system, "motion", extension)


def make_channels_filename(
    subject_id: str,
    session_label: str,
    task_name: str,
    system: TrackingSystem,
    extension: str = "tsv",
) -> str:
    """
    Generate BIDS channels descriptor filename.

    Example: sub-01_ses-01_task-VRtracking_tracksys-Head_channels.tsv
    """
    return make_bids_filename(subject_id, session_label, task_name, system, "channels", extension)
