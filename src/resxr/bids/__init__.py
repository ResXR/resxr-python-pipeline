"""
BIDS module for ResXR pipeline.

Handles BIDS directory structure, naming conventions, and metadata generation.
"""

from .channels import generate_channels_tsv
from .layout import BIDSLayout
from .metadata import (
    generate_channels_json,
    generate_dataset_description,
    generate_derivative_description,
    generate_motion_json,
    generate_participants_json,
)
from .naming import (
    make_bids_filename,
    make_channels_filename,
    make_motion_filename,
)

__all__ = [
    "BIDSLayout",
    "generate_motion_json",
    "generate_channels_json",
    "generate_dataset_description",
    "generate_derivative_description",
    "generate_participants_json",
    "generate_channels_tsv",
    "make_bids_filename",
    "make_motion_filename",
    "make_channels_filename",
]
