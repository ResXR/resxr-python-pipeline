"""
BIDS metadata (sidecar JSON) generation for ResXR pipeline.

Generates ``_motion.json`` sidecar files.  When ``prepared_data`` is
provided (the output of ``prepare_motion_data``), channel counts and
column lists are derived from it — ensuring the JSON, TSV, and
channels files all describe the same set of columns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.config import BIDSConfig, DeviceConfig
from ..core.session import Session, TrackingStream
from ..io.column_maps import count_channel_types, count_tracked_points


def generate_motion_json(
    stream: TrackingStream,
    session: Session,
    task_name: str,
    device: DeviceConfig,
    bids_config: BIDSConfig,
    system_descriptions: dict[str, str],
    prepared_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Generate _motion.json sidecar metadata for a tracking stream.

    Parameters
    ----------
    stream : TrackingStream
        Tracking stream to generate metadata for
    session : Session
        Session containing the stream
    task_name : str
        BIDS task name (from config)
    device : DeviceConfig
        Device configuration for manufacturer/model metadata (from config)
    bids_config : BIDSConfig
        BIDS specification configuration (from config)
    system_descriptions : Dict[str, str]
        Mapping of system names to task descriptions (from config)
    prepared_data : pd.DataFrame | None
        Pre-prepared BIDS-ready DataFrame (output of prepare_motion_data).
        If None, falls back to stream.get_output_data() with internal
        column exclusion.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata dictionary
    """
    if prepared_data is not None:
        data_cols = list(prepared_data.columns)
    else:
        # Fallback: read from stream and exclude internal time columns
        data = stream.get_output_data()
        exclude = {"timestamp", "timeSinceStartup"}
        data_cols = [c for c in data.columns if c not in exclude]

    # Count channels by type
    channel_counts = count_channel_types(data_cols)

    # Count tracked points
    tracked_points = count_tracked_points(data_cols)

    # Build software version string. Folds the session's OS/runtime strings into
    # the standard BIDS SoftwareVersions field (BIDS has no dedicated OS field).
    software_versions = []
    if session.metadata.unity_version:
        software_versions.append(f"Unity {session.metadata.unity_version}")
    if session.metadata.ovrplugin_version:
        software_versions.append(f"OVR Plugin v{session.metadata.ovrplugin_version}")
    # Meta Horizon OS release; skip Play Mode / PCVR / read-failure sentinels.
    horizon_os = session.metadata.horizon_os_version
    if horizon_os and horizon_os not in {"editor", "n/a", "unknown"}:
        software_versions.append(f"Horizon OS {horizon_os}")
    # Full Android OS / build string, if captured.
    if session.metadata.software_versions_raw:
        software_versions.append(session.metadata.software_versions_raw)
    software_str = ", ".join(software_versions) if software_versions else "n/a"

    # Determine task description: use config override, or fall back to system description
    if device.task_description:
        task_description = device.task_description
    else:
        task_description = system_descriptions.get(
            stream.system.value,
            f"{stream.system.value} tracking from {device.model_name} VR headset",
        )

    metadata = {
        "TaskName": task_name,
        "TaskDescription": task_description,
        "Manufacturer": device.manufacturer,
        "ManufacturersModelName": device.model_name,
        "SoftwareVersions": software_str,
        "TrackingSystemName": stream.system.value,
        "SamplingFrequency": stream.sampling_frequency,
        "SamplingFrequencyEffective": stream.sampling_frequency_effective
        if stream.sampling_frequency_effective > 0
        else "n/a",
        "MissingValues": bids_config.missing_values,
        "MotionChannelCount": len(data_cols),
        "TrackedPointsCount": tracked_points,
    }

    # DeviceSerialNumber is RECOMMENDED but typically unavailable: Meta blocks
    # serial reads on Android 10+. Emit only when known; BIDS prefers omitting an
    # unknown optional field over writing an empty string.
    if session.metadata.device_serial_number:
        metadata["DeviceSerialNumber"] = session.metadata.device_serial_number

    # Add channel type counts
    metadata.update(channel_counts)

    return metadata


def generate_channels_json(bids_config: BIDSConfig) -> dict[str, Any]:
    """
    Generate _channels.json sidecar with reference frame specification.

    Parameters
    ----------
    bids_config : BIDSConfig
        BIDS specification configuration (from config)

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata dictionary
    """
    ref = bids_config.reference_frame
    return {
        "reference_frame": {
            "Description": ref.description,
            "Levels": {
                "global": {
                    "Description": ref.description,
                    "RotationRule": ref.rotation_rule,
                    "RotationOrder": ref.rotation_order,
                    "SpatialAxes": ref.spatial_axes,
                }
            },
        }
    }


def generate_dataset_description(
    dataset_name: str,
    bids_version: str,
    bids_config: BIDSConfig,
) -> dict[str, Any]:
    """
    Generate dataset_description.json content.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    bids_version : str
        BIDS specification version
    bids_config : BIDSConfig
        BIDS specification configuration (from config)

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata dictionary
    """
    return {
        "Name": dataset_name,
        "BIDSVersion": bids_version,
        "DatasetType": bids_config.dataset_type,
        "License": bids_config.license,
        "Authors": bids_config.authors,
    }


def generate_derivative_description(
    dataset_name: str,
    bids_version: str,
) -> dict[str, Any]:
    """
    Generate dataset_description.json for a BIDS derivative dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    bids_version : str
        BIDS specification version

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata dictionary
    """
    return {
        "Name": f"{dataset_name} (preprocessed)",
        "BIDSVersion": bids_version,
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "ResXR",
            }
        ],
    }


def generate_participants_json() -> dict[str, Any]:
    """
    Generate participants.json column definitions.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable metadata dictionary
    """
    return {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Age of the participant", "Units": "years"},
        "sex": {"Description": "Sex of the participant", "Levels": {"M": "male", "F": "female"}},
        "handedness": {
            "Description": "Handedness of the participant",
            "Levels": {"R": "right-handed", "L": "left-handed", "A": "ambidextrous"},
        },
    }
