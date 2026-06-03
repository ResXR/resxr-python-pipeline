"""Tests for PipelineConfig and all config dataclasses (core/config.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from resxr.core.config import (
    BIDSConfig,
    ColumnGroup,
    DeviceConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    ReferenceFrameConfig,
    ReportConfig,
    SessionMapping,
    ValidationConfig,
)
from resxr.core.exceptions import ConfigurationError

# ===========================================================================
# PipelineConfig.from_yaml
# ===========================================================================


class TestFromYaml:
    def test_valid_config_loads_successfully(self, tmp_config_yaml):
        """A valid YAML file produces a PipelineConfig without errors."""
        cfg = PipelineConfig.from_yaml(tmp_config_yaml)
        assert isinstance(cfg, PipelineConfig)

    def test_nonexistent_file_raises_configuration_error(self, tmp_path):
        """Passing a nonexistent path raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="not found"):
            PipelineConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        """A file with invalid YAML syntax raises ConfigurationError."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed\n  bad: yaml\n")
        with pytest.raises(ConfigurationError, match="parse"):
            PipelineConfig.from_yaml(bad)

    def test_empty_yaml_raises_configuration_error(self, tmp_path):
        """An empty YAML file raises ConfigurationError."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ConfigurationError):
            PipelineConfig.from_yaml(empty)

    def test_missing_input_section_raises(self, tmp_path, minimal_config_dict):
        """Missing required 'input' section raises ConfigurationError."""
        del minimal_config_dict["input"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match="input"):
            PipelineConfig.from_yaml(path)

    def test_missing_output_section_raises(self, tmp_path, minimal_config_dict):
        """Missing required 'output' section raises ConfigurationError."""
        del minimal_config_dict["output"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match="output"):
            PipelineConfig.from_yaml(path)

    def test_missing_validation_section_raises(self, tmp_path, minimal_config_dict):
        """Missing required 'validation' section raises ConfigurationError."""
        del minimal_config_dict["validation"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match="validation"):
            PipelineConfig.from_yaml(path)

    def test_missing_bids_section_raises(self, tmp_path, minimal_config_dict):
        """Missing required 'bids' section raises ConfigurationError."""
        del minimal_config_dict["bids"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match="bids"):
            PipelineConfig.from_yaml(path)

    def test_missing_sampling_frequencies_raises(self, tmp_path, minimal_config_dict):
        """Missing sampling_frequencies section raises ConfigurationError."""
        del minimal_config_dict["sampling_frequencies"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match="sampling_frequencies"):
            PipelineConfig.from_yaml(path)

    def test_wrong_type_enabled_checks_raises_configuration_error(
        self, tmp_path, minimal_config_dict
    ):
        """Non-list validation.enabled_checks raises a clear ConfigurationError."""
        minimal_config_dict["validation"]["enabled_checks"] = "not_a_list"
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match=r"enabled_checks"):
            PipelineConfig.from_yaml(path)

    def test_wrong_type_sampling_frequencies_raises_configuration_error(
        self, tmp_path, minimal_config_dict
    ):
        """Non-dict sampling_frequencies raises a clear ConfigurationError."""
        minimal_config_dict["sampling_frequencies"] = "not_a_dict"
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match=r"sampling_frequencies"):
            PipelineConfig.from_yaml(path)

    def test_duplicate_column_group_name_raises(self, tmp_path, minimal_config_dict):
        """Duplicate column_group names raise ConfigurationError."""
        minimal_config_dict["validation"]["column_groups"] = [
            {"name": "GroupA", "description": "d", "columns": ["col1"]},
            {"name": "GroupA", "description": "d", "columns": ["col2"]},
        ]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        with pytest.raises(ConfigurationError, match="Duplicate"):
            PipelineConfig.from_yaml(path)


class TestConfigFields:
    def test_input_config_fields(self, pipeline_config):
        """InputConfig fields are accessible and correctly typed."""
        ic = pipeline_config.input
        assert isinstance(ic, InputConfig)
        assert isinstance(ic.data_dir, Path)
        assert isinstance(ic.continuous_data_pattern, str)
        assert isinstance(ic.face_data_pattern, str)
        assert isinstance(ic.metadata_pattern, str)
        assert isinstance(ic.events_data_pattern, str)


def test_input_config_custom_tables_dir_default(minimal_config_dict):
    from resxr.core.config import PipelineConfig

    cfg = PipelineConfig.model_validate(minimal_config_dict)
    assert cfg.input.custom_tables_dir == "custom_tables"


def test_input_config_custom_tables_dir_parsed(minimal_config_dict):
    from resxr.core.config import PipelineConfig

    d = dict(minimal_config_dict)
    d["input"] = dict(d["input"])
    d["input"]["custom_tables_dir"] = "events_extra"
    cfg = PipelineConfig.model_validate(d)
    assert cfg.input.custom_tables_dir == "events_extra"

    def test_output_config_fields(self, pipeline_config):
        """OutputConfig fields are accessible and correctly typed."""
        oc = pipeline_config.output
        assert isinstance(oc, OutputConfig)
        assert isinstance(oc.bids_root, Path)
        assert isinstance(oc.dataset_name, str)
        assert isinstance(oc.task_name, str)
        assert isinstance(oc.overwrite, bool)

    def test_device_config_fields(self, pipeline_config):
        """DeviceConfig fields are accessible."""
        dc = pipeline_config.device
        assert isinstance(dc, DeviceConfig)
        assert dc.manufacturer == "Meta"
        assert dc.model_name == "Meta Quest Pro"

    def test_bids_config_fields(self, pipeline_config):
        """BIDSConfig fields are accessible."""
        bc = pipeline_config.bids
        assert isinstance(bc, BIDSConfig)
        assert isinstance(bc.missing_values, str)
        assert isinstance(bc.license, str)
        assert isinstance(bc.authors, list)

    def test_reference_frame_config_fields(self, pipeline_config):
        """ReferenceFrameConfig fields are strings."""
        rf = pipeline_config.bids.reference_frame
        assert isinstance(rf, ReferenceFrameConfig)
        assert isinstance(rf.description, str)
        assert isinstance(rf.rotation_rule, str)
        assert isinstance(rf.rotation_order, str)
        assert isinstance(rf.spatial_axes, str)

    def test_sampling_frequencies_dict(self, pipeline_config):
        """sampling_frequencies is a dict mapping system names to floats."""
        sf = pipeline_config.sampling_frequencies
        assert isinstance(sf, dict)
        assert "Head" in sf
        assert sf["Head"] == pytest.approx(90.0)

    def test_preprocessing_config_fields(self, pipeline_config):
        """PreprocessingConfig is accessible with expected defaults."""
        pc = pipeline_config.preprocessing
        assert isinstance(pc, PreprocessingConfig)
        assert isinstance(pc.apply_quality_masking, bool)

    def test_report_config_fields(self, pipeline_config):
        """ReportConfig enabled flag is accessible."""
        rc = pipeline_config.report
        assert isinstance(rc, ReportConfig)
        assert isinstance(rc.enabled, bool)

    def test_session_mappings_optional(self, pipeline_config):
        """session_mappings is a list (may be empty)."""
        assert isinstance(pipeline_config.session_mappings, list)

    def test_session_mapping_fields(self, tmp_path, minimal_config_dict):
        """SessionMapping fields are accessible after loading from YAML."""
        minimal_config_dict["session_mappings"] = [
            {
                "source_dir": "session_001",
                "subject_id": "01",
                "session_label": "01",
                "age": "25",
                "sex": "M",
                "handedness": "R",
            }
        ]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(minimal_config_dict))
        cfg = PipelineConfig.from_yaml(path)
        m = cfg.session_mappings[0]
        assert isinstance(m, SessionMapping)
        assert m.source_dir == "session_001"
        assert m.subject_id == "01"
        assert m.session_label == "01"
        assert m.age == "25"
        assert m.sex == "M"
        assert m.handedness == "R"


# ===========================================================================
# ValidationConfig.get_column_groups
# ===========================================================================


class TestGetColumnGroups:
    def _make_validation_config_with_groups(self) -> ValidationConfig:
        return ValidationConfig(
            enabled_checks=["hands_tracking_loss", "sampling_rate"],
            column_groups=[
                ColumnGroup(name="Left Hand", description="Left", columns=["LeftHand_Root_px"]),
                ColumnGroup(name="Right Hand", description="Right", columns=["RightHand_Root_px"]),
            ],
            settings={
                "check_column_groups": {
                    "hands_tracking_loss": ["Left Hand"],
                }
            },
        )

    def test_no_filter_returns_all_groups(self):
        """get_column_groups() with no argument returns all configured groups."""
        vc = self._make_validation_config_with_groups()
        groups = vc.get_column_groups()
        assert len(groups) == 2
        names = [g.name for g in groups]
        assert "Left Hand" in names
        assert "Right Hand" in names

    def test_filter_by_check_name_returns_subset(self):
        """get_column_groups(check_name=...) returns only groups assigned to that check."""
        vc = self._make_validation_config_with_groups()
        groups = vc.get_column_groups(check_name="hands_tracking_loss")
        assert len(groups) == 1
        assert groups[0].name == "Left Hand"

    def test_filter_unknown_check_returns_all(self):
        """check_name not in check_column_groups → all groups returned."""
        vc = self._make_validation_config_with_groups()
        groups = vc.get_column_groups(check_name="unknown_check")
        assert len(groups) == 2

    def test_no_groups_configured_returns_empty(self):
        """Empty column_groups with no default_columns returns []."""
        vc = ValidationConfig(enabled_checks=[], column_groups=[])
        assert vc.get_column_groups() == []

    def test_no_groups_with_default_columns(self):
        """Empty column_groups + default_columns returns a single 'All Columns' group."""
        vc = ValidationConfig(enabled_checks=[], column_groups=[])
        groups = vc.get_column_groups(default_columns=["col1", "col2"])
        assert len(groups) == 1
        assert groups[0].name == "All Columns"
        assert groups[0].columns == ["col1", "col2"]

    def test_get_settings_via_get_method(self):
        """ValidationConfig.get() retrieves values from settings dict."""
        vc = ValidationConfig(
            enabled_checks=[],
            settings={"my_threshold": 0.95},
        )
        assert vc.get("my_threshold") == pytest.approx(0.95)
        assert vc.get("nonexistent", "default") == "default"
