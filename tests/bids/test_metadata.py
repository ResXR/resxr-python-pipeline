"""Tests for BIDS metadata/JSON sidecar generators (bids/metadata.py)."""

from __future__ import annotations

import pytest

from resxr.bids.metadata import (
    generate_channels_json,
    generate_dataset_description,
    generate_derivative_description,
    generate_motion_json,
    generate_participants_json,
)
from resxr.core.config import BIDSConfig, DeviceConfig, ReferenceFrameConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bids_config() -> BIDSConfig:
    return BIDSConfig(
        missing_values="n/a",
        dataset_type="raw",
        license="CC0",
        authors=["Tester"],
        reference_frame=ReferenceFrameConfig(
            description="Right-hand",
            rotation_rule="right-hand",
            rotation_order="XYZ",
            spatial_axes="RAS",
        ),
    )


def _device_config() -> DeviceConfig:
    return DeviceConfig(manufacturer="Meta", model_name="Meta Quest Pro")


# ===========================================================================
# generate_motion_json
# ===========================================================================


class TestGenerateMotionJson:
    def test_returns_dict(self, head_stream, full_session, pipeline_config):
        """generate_motion_json returns a dict."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={"Head": "Head tracking"},
        )
        assert isinstance(result, dict)

    def test_task_name_present(self, head_stream, full_session):
        """TaskName field is set from the provided task_name argument."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="my_task",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["TaskName"] == "my_task"

    def test_manufacturer_from_device_config(self, head_stream, full_session):
        """Manufacturer field comes from device config."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["Manufacturer"] == "Meta"

    def test_model_name_from_device_config(self, head_stream, full_session):
        """ManufacturersModelName field comes from device config."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["ManufacturersModelName"] == "Meta Quest Pro"

    def test_sampling_frequency_matches_stream(self, head_stream, full_session):
        """SamplingFrequency equals stream.sampling_frequency."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["SamplingFrequency"] == pytest.approx(head_stream.sampling_frequency)

    def test_tracking_system_name(self, head_stream, full_session):
        """TrackingSystemName is the system's .value string."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["TrackingSystemName"] == "Head"

    def test_missing_values_from_bids_config(self, head_stream, full_session):
        """MissingValues field is set from bids_config."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["MissingValues"] == "n/a"

    def test_motion_channel_count_present(self, head_stream, full_session):
        """MotionChannelCount key is present."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert "MotionChannelCount" in result
        assert isinstance(result["MotionChannelCount"], int)

    def test_prepared_data_overrides_stream_columns(self, head_stream, full_session):
        """When prepared_data is provided its columns are used for counting."""
        import pandas as pd

        prepared = pd.DataFrame(
            {
                "latency": [0.0, 0.1],
                "Node_Head_px": [0.0, 0.1],
            }
        )
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
            prepared_data=prepared,
        )
        assert result["MotionChannelCount"] == 2

    def test_motion_channel_count_matches_stream_columns(self, head_stream, full_session):
        """MotionChannelCount equals the number of data columns in the stream output."""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        # head_stream output data excludes timestamp/timeSinceStartup
        output_data = head_stream.get_output_data()
        exclude = {"timestamp", "timeSinceStartup"}
        expected_cols = [c for c in output_data.columns if c not in exclude]
        assert result["MotionChannelCount"] == len(expected_cols)

    def test_task_description_overridden_by_device_config(self, head_stream, full_session):
        """device.task_description takes priority over system_descriptions."""
        device = DeviceConfig(
            manufacturer="Meta",
            model_name="Quest Pro",
            task_description="My custom description",
        )
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=device,
            bids_config=_bids_config(),
            system_descriptions={"Head": "Head tracking"},
        )
        assert result["TaskDescription"] == "My custom description"

    def test_software_versions_folds_in_horizon_os(self, head_stream, full_session):
        """horizon_os_version is appended to the SoftwareVersions string."""
        full_session.metadata.horizon_os_version = "2.4"
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert "Horizon OS 2.4" in result["SoftwareVersions"]

    @pytest.mark.parametrize("sentinel", ["editor", "n/a", "unknown"])
    def test_software_versions_skips_horizon_sentinels(
        self, sentinel, head_stream, full_session
    ):
        """Editor / PCVR / read-failure sentinels are not folded into SoftwareVersions."""
        full_session.metadata.horizon_os_version = sentinel
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert "Horizon OS" not in result["SoftwareVersions"]

    def test_software_versions_folds_in_raw_os_string(self, head_stream, full_session):
        """software_versions_raw (full Android/build string) is appended verbatim."""
        full_session.metadata.software_versions_raw = "Android OS 14 / API-34"
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert "Android OS 14 / API-34" in result["SoftwareVersions"]

    def test_device_serial_number_omitted_when_empty(self, head_stream, full_session):
        """DeviceSerialNumber is omitted entirely when not captured (BIDS convention)."""
        full_session.metadata.device_serial_number = ""
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert "DeviceSerialNumber" not in result

    def test_device_serial_number_present_when_known(self, head_stream, full_session):
        """DeviceSerialNumber is emitted when the session carries one."""
        full_session.metadata.device_serial_number = "SN-ABC-123"
        result = generate_motion_json(
            stream=head_stream,
            session=full_session,
            task_name="vr",
            device=_device_config(),
            bids_config=_bids_config(),
            system_descriptions={},
        )
        assert result["DeviceSerialNumber"] == "SN-ABC-123"


# ===========================================================================
# generate_dataset_description
# ===========================================================================


class TestGenerateDatasetDescription:
    def test_returns_dict(self):
        """Returns a dict."""
        result = generate_dataset_description("TestDS", "1.10.1", _bids_config())
        assert isinstance(result, dict)

    def test_name_field(self):
        """Name field matches dataset_name argument."""
        result = generate_dataset_description("MyDataset", "1.10.1", _bids_config())
        assert result["Name"] == "MyDataset"

    def test_bids_version_field(self):
        """BIDSVersion field matches bids_version argument."""
        result = generate_dataset_description("DS", "1.10.1", _bids_config())
        assert result["BIDSVersion"] == "1.10.1"

    def test_license_from_bids_config(self):
        """License field comes from bids_config."""
        result = generate_dataset_description("DS", "1.10.1", _bids_config())
        assert result["License"] == "CC0"

    def test_authors_from_bids_config(self):
        """Authors list comes from bids_config."""
        result = generate_dataset_description("DS", "1.10.1", _bids_config())
        assert result["Authors"] == ["Tester"]

    def test_required_keys_present(self):
        """Name, BIDSVersion, DatasetType, License, Authors are all present."""
        result = generate_dataset_description("DS", "1.10.1", _bids_config())
        for key in ("Name", "BIDSVersion", "DatasetType", "License", "Authors"):
            assert key in result


# ===========================================================================
# generate_derivative_description
# ===========================================================================


class TestGenerateDerivativeDescription:
    def test_returns_dict(self):
        """Returns a dict."""
        result = generate_derivative_description("DS", "1.10.1")
        assert isinstance(result, dict)

    def test_dataset_type_is_derivative(self):
        """DatasetType is 'derivative'."""
        result = generate_derivative_description("DS", "1.10.1")
        assert result["DatasetType"] == "derivative"

    def test_generated_by_field_present(self):
        """GeneratedBy key is present."""
        result = generate_derivative_description("DS", "1.10.1")
        assert "GeneratedBy" in result

    def test_name_based_on_input(self):
        """Name field is derived from dataset_name."""
        result = generate_derivative_description("TestDS", "1.10.1")
        assert "TestDS" in result["Name"]


# ===========================================================================
# generate_participants_json
# ===========================================================================


class TestGenerateParticipantsJson:
    def test_returns_dict(self):
        """Returns a dict."""
        result = generate_participants_json()
        assert isinstance(result, dict)

    def test_participant_id_key_present(self):
        """'participant_id' key is present."""
        assert "participant_id" in generate_participants_json()

    def test_standard_bids_keys_present(self):
        """age, sex, handedness keys are present."""
        result = generate_participants_json()
        for key in ("age", "sex", "handedness"):
            assert key in result


# ===========================================================================
# generate_channels_json (reference frame propagation)
# ===========================================================================


class TestGenerateChannelsJson:
    def test_returns_dict(self):
        """generate_channels_json returns a dict."""
        result = generate_channels_json(_bids_config())
        assert isinstance(result, dict)

    def test_reference_frame_key_present(self):
        """Result contains a 'reference_frame' key."""
        result = generate_channels_json(_bids_config())
        assert "reference_frame" in result

    def test_reference_frame_description_propagated(self):
        """reference_frame.Description matches bids_config.reference_frame.description."""
        result = generate_channels_json(_bids_config())
        assert result["reference_frame"]["Description"] == "Right-hand"

    def test_rotation_rule_propagated(self):
        """RotationRule from bids_config.reference_frame is passed through."""
        result = generate_channels_json(_bids_config())
        global_level = result["reference_frame"]["Levels"]["global"]
        assert global_level["RotationRule"] == "right-hand"

    def test_rotation_order_propagated(self):
        """RotationOrder from bids_config.reference_frame is passed through."""
        result = generate_channels_json(_bids_config())
        global_level = result["reference_frame"]["Levels"]["global"]
        assert global_level["RotationOrder"] == "XYZ"

    def test_spatial_axes_propagated(self):
        """SpatialAxes from bids_config.reference_frame is passed through."""
        result = generate_channels_json(_bids_config())
        global_level = result["reference_frame"]["Levels"]["global"]
        assert global_level["SpatialAxes"] == "RAS"
