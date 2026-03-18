"""
BIDS directory layout management for ResXR pipeline.

Creates and manages BIDS-compliant directory structures.
"""

from __future__ import annotations

from pathlib import Path

from ..core.constants import TrackingSystem
from ..core.logger import get_logger
from ..core.session import Session
from .naming import make_channels_filename, make_motion_filename

logger = get_logger(__name__)


class BIDSLayout:
    """
    Manages BIDS directory structure creation and path generation.

    Attributes
    ----------
    bids_root : Path
        Root directory of the BIDS dataset
    task_name : str
        Task name used in filenames
    """

    def __init__(self, bids_root: Path, task_name: str):
        """
        Initialize BIDS layout manager.

        Parameters
        ----------
        bids_root : Path
            Root directory for BIDS output
        task_name : str
            Task name for BIDS filenames (from config)
        """
        self.bids_root = Path(bids_root)
        self.task_name = task_name

    def _get_base_root(self, derivative: bool = False) -> Path:
        """
        Get base root directory (raw or derivative).

        Parameters
        ----------
        derivative : bool
            If True, return path to derivatives/resxr subdirectory

        Returns
        -------
        Path
            Base root path
        """
        if derivative:
            return self.bids_root / "derivatives" / "resxr"
        return self.bids_root

    def create_structure(self, session: Session, derivative: bool = False) -> None:
        """
        Create BIDS directory structure for a session.

        Parameters
        ----------
        session : Session
            Session to create directories for
        derivative : bool
            If True, create derivative directory structure
        """
        motion_dir = self.get_motion_dir(session, derivative)
        motion_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {motion_dir}")

    def get_subject_dir(self, session: Session, derivative: bool = False) -> Path:
        """Get path to subject directory."""
        return self._get_base_root(derivative) / f"sub-{session.subject_id}"

    def get_session_dir(self, session: Session, derivative: bool = False) -> Path:
        """Get path to session directory."""
        return self.get_subject_dir(session, derivative) / f"ses-{session.session_label}"

    def get_motion_dir(self, session: Session, derivative: bool = False) -> Path:
        """Get path to motion data directory."""
        return self.get_session_dir(session, derivative) / "motion"

    def get_motion_file(
        self,
        session: Session,
        system: TrackingSystem,
        extension: str = "tsv",
        derivative: bool = False,
    ) -> Path:
        """
        Get path to motion data file.

        Parameters
        ----------
        session : Session
            Session object
        system : TrackingSystem
            Tracking system
        extension : str
            File extension ("tsv" or "json")
        derivative : bool
            If True, return path under derivatives/resxr

        Returns
        -------
        Path
            Full path to motion file
        """
        filename = make_motion_filename(
            session.subject_id, session.session_label, self.task_name, system, extension
        )
        return self.get_motion_dir(session, derivative) / filename

    def get_channels_file(
        self,
        session: Session,
        system: TrackingSystem,
        extension: str = "tsv",
        derivative: bool = False,
    ) -> Path:
        """
        Get path to channels descriptor file.

        Parameters
        ----------
        session : Session
            Session object
        system : TrackingSystem
            Tracking system
        extension : str
            File extension ("tsv" or "json")
        derivative : bool
            If True, return path under derivatives/resxr

        Returns
        -------
        Path
            Full path to channels file
        """
        filename = make_channels_filename(
            session.subject_id, session.session_label, self.task_name, system, extension
        )
        return self.get_motion_dir(session, derivative) / filename

    def get_scans_file(self, session: Session, derivative: bool = False) -> Path:
        """Get path to scans.tsv file."""
        filename = f"sub-{session.subject_id}_ses-{session.session_label}_scans.tsv"
        return self.get_session_dir(session, derivative) / filename

    def get_dataset_description_file(self, derivative: bool = False) -> Path:
        """Get path to dataset_description.json."""
        return self._get_base_root(derivative) / "dataset_description.json"

    def get_participants_file(self, extension: str = "tsv") -> Path:
        """Get path to participants file."""
        return self.bids_root / f"participants.{extension}"

    def get_bidsignore_file(self) -> Path:
        """Get path to .bidsignore file."""
        return self.bids_root / ".bidsignore"
