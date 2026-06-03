"""
Configuration management for the ResXR pipeline.

Provides Pydantic models for type-safe configuration and YAML loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)

from .exceptions import ConfigurationError
from .logger import get_logger

_config_logger = get_logger(__name__)


class ConfigModel(BaseModel):
    """Base model for pipeline configuration objects."""

    model_config = ConfigDict(extra="ignore")


class InputConfig(ConfigModel):
    """Configuration for input data sources."""

    data_dir: Path
    continuous_data_pattern: str
    face_data_pattern: str
    metadata_pattern: str
    events_data_pattern: str
    # Subfolder within each source dir holding custom_tables.json and the custom
    # data-class CSVs. Every CSV there is loaded as a custom class.
    custom_tables_dir: str = "custom_tables"


class OutputConfig(ConfigModel):
    """Configuration for BIDS output."""

    bids_root: Path
    dataset_name: str
    bids_version: str
    task_name: str
    overwrite: StrictBool


class SessionMapping(ConfigModel):
    """Maps source directory to BIDS subject/session identifiers."""

    source_dir: str
    subject_id: str
    session_label: str
    age: str | None = None  # BIDS participants age; None -> "n/a"
    sex: str | None = None  # BIDS participants sex; None -> "n/a"
    handedness: str | None = None  # BIDS participants handedness; None -> "n/a"


class TrackingSystemConfig(ConfigModel):
    """Per-tracking-system configuration."""

    enabled: StrictBool


class DeviceConfig(ConfigModel):
    """Configuration for device/hardware metadata in BIDS output."""

    manufacturer: str
    model_name: str
    task_description: str | None = None  # None means use system defaults


class ColumnGroup(ConfigModel):
    """A named group of columns for column-scoped validation checks."""

    name: str
    description: str = ""
    columns: list[str]


class ValidationConfig(ConfigModel):
    """Configuration for quality validation checks."""

    enabled_checks: list[str]
    column_groups: list[ColumnGroup] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_duplicate_group_names(self) -> ValidationConfig:
        """Reject duplicate column group names with a clear config error."""
        seen_names: set[str] = set()
        for group in self.column_groups:
            if group.name in seen_names:
                raise ValueError(f"Duplicate column_groups name: {group.name}")
            seen_names.add(group.name)
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from settings, with an optional default."""
        return self.settings.get(key, default)

    def get_column_groups(
        self,
        check_name: str | None = None,
        default_columns: list[str] | None = None,
    ) -> list[ColumnGroup]:
        """Get column groups, optionally filtered for a specific check."""
        if not self.column_groups:
            if default_columns is not None:
                return [
                    ColumnGroup(
                        name="All Columns",
                        description="Default — all columns",
                        columns=default_columns,
                    )
                ]
            return []

        if check_name is None:
            return list(self.column_groups)

        selected = (self.get("check_column_groups") or {}).get(check_name)
        if selected is None:
            return list(self.column_groups)

        by_name = {group.name: group for group in self.column_groups}
        missing = [n for n in selected if n not in by_name]
        if missing:
            raise ConfigurationError(
                f"check_column_groups references unknown group(s): {missing}. "
                f"Available: {list(by_name)}"
            )
        return [by_name[name] for name in selected]


class PreprocessingConfig(ConfigModel):
    """Configuration for preprocessing steps."""

    apply_quality_masking: StrictBool = False
    masking_checks: list[str] | None = None
    alternate_time_columns: dict[str, str] = Field(default_factory=dict)


class ReportConfig(ConfigModel):
    """Configuration for HTML report generation (dashboard)."""

    enabled: StrictBool
    output_dir: Path | None = None


class ReferenceFrameConfig(ConfigModel):
    """Configuration for BIDS coordinate system reference frame."""

    description: str
    rotation_rule: str
    rotation_order: str
    spatial_axes: str


class BIDSConfig(ConfigModel):
    """Configuration for BIDS specification values."""

    missing_values: str
    dataset_type: str
    license: str
    authors: list[str] = Field(default_factory=list)
    reference_frame: ReferenceFrameConfig


class PipelineConfig(ConfigModel):
    """Complete pipeline configuration."""

    input: InputConfig
    output: OutputConfig
    device: DeviceConfig
    validation: ValidationConfig
    preprocessing: PreprocessingConfig
    report: ReportConfig
    bids: BIDSConfig
    sampling_frequencies: dict[str, float]
    system_descriptions: dict[str, str]
    session_mappings: list[SessionMapping] = Field(default_factory=list)
    systems: dict[str, TrackingSystemConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : Path
            Path to the YAML configuration file

        Returns
        -------
        PipelineConfig
            Parsed configuration object

        Raises
        ------
        ConfigurationError
            If the file cannot be read, parsed, or is missing required fields
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ConfigurationError(f"Failed to read config: {e}") from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML config: {e}") from e

        if not data:
            raise ConfigurationError("Empty configuration")

        known_keys = set(cls.model_fields.keys())
        if isinstance(data, dict):
            unknown = sorted(set(data.keys()) - known_keys)
            if unknown:
                _config_logger.warning(
                    "Unknown top-level config keys (ignored): %s. Check for typos. Known keys: %s",
                    unknown,
                    sorted(known_keys),
                )

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ConfigurationError(str(e)) from e

    def is_system_enabled(self, system_name: str) -> bool:
        """Check if a tracking system is enabled in config."""
        if system_name not in self.systems:
            return True  # Default to enabled if not specified
        return self.systems[system_name].enabled
