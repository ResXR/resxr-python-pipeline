"""Tests for the ResXR command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from resxr.cli import build_parser, main
from resxr.core.config import PipelineConfig
from resxr.core.exceptions import ConfigurationError
from resxr.pipeline import validate_config

_MASKING_NOT_ENABLED = (
    "preprocessing.masking_checks names check(s) {names} that are not in "
    "validation.enabled_checks; they never run, so nothing would be masked for them. "
    "Add them to enabled_checks or remove them from masking_checks."
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for CLI argument parser construction."""

    def test_default_config_path(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.config == "config/pipeline_config.yaml"

    def test_custom_config_path(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "my/config.yaml"])
        assert args.config == "my/config.yaml"

    def test_long_config_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--config", "other.yaml"])
        assert args.config == "other.yaml"

    def test_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_quiet_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-q"])
        assert args.quiet is True

    def test_defaults_are_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.dry_run is False
        assert args.verbose is False
        assert args.quiet is False


# ---------------------------------------------------------------------------
# main() — config not found
# ---------------------------------------------------------------------------


class TestMainConfigNotFound:
    """main() should return exit code 1 when config file is missing."""

    def test_missing_config_returns_1(self, tmp_path):
        exit_code = main(["-c", str(tmp_path / "nonexistent.yaml")])
        assert exit_code == 1


# ---------------------------------------------------------------------------
# main() — dry-run mode
# ---------------------------------------------------------------------------


class TestMainDryRun:
    """main() --dry-run validates config and returns 0 without writing output."""

    @pytest.fixture()
    def valid_config_yaml(self, tmp_path, minimal_config_dict) -> Path:
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(minimal_config_dict, f)
        return path

    def test_dry_run_valid_config_returns_0(self, valid_config_yaml):
        exit_code = main(["--config", str(valid_config_yaml), "--dry-run"])
        assert exit_code == 0

    def test_dry_run_invalid_config_returns_1(self, tmp_path):
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("not_a_valid_key: true\n")
        exit_code = main(["--config", str(bad_config), "--dry-run"])
        assert exit_code == 1

    def test_dry_run_does_not_write_output(self, valid_config_yaml, tmp_path):
        out_dir = tmp_path / "output"
        main(["--config", str(valid_config_yaml), "--dry-run"])
        assert not out_dir.exists()

    def test_dry_run_rejects_misspelled_check_name(self, tmp_path, minimal_config_dict, caplog):
        minimal_config_dict["validation"]["enabled_checks"] = ["hand_tracking_loss"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert "validation.enabled_checks names unknown check(s) ['hand_tracking_loss']" in (
            caplog.text
        )
        assert "hands_tracking_loss" in caplog.text  # the valid names are listed

    def test_dry_run_rejects_a_masking_check_that_is_not_enabled(
        self, tmp_path, minimal_config_dict, caplog
    ):
        minimal_config_dict["preprocessing"].update(
            apply_quality_masking=True, masking_checks=["eyes_closed"]
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert "Configuration error: " in caplog.text
        assert _MASKING_NOT_ENABLED.format(names="['eyes_closed']") in caplog.text


# ---------------------------------------------------------------------------
# main() — configuration errors stop the run before any session
# ---------------------------------------------------------------------------


class TestMainConfigErrors:
    """A configuration error exits 1 before any session is read or written."""

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            (
                lambda cfg: cfg["validation"].update(enabled_checks=["hand_tracking_loss"]),
                "validation.enabled_checks names unknown check(s) ['hand_tracking_loss']",
            ),
            (
                lambda cfg: cfg["preprocessing"].update(masking_checks=["eye_closed"]),
                "preprocessing.masking_checks names unknown check(s) ['eye_closed']",
            ),
            (
                lambda cfg: cfg.update(systems={"Hand": {"enabled": False}}),
                "systems names unknown tracking system(s) ['Hand']",
            ),
            (
                lambda cfg: cfg["preprocessing"].update(
                    alternate_time_columns={"Eye": "Eyes_Time"}
                ),
                "preprocessing.alternate_time_columns names unknown tracking system(s) ['Eye']",
            ),
            (
                lambda cfg: cfg["sampling_frequencies"].update(Hands=0.0),
                "sampling_frequencies must be positive; got {'Hands': 0.0}",
            ),
            (
                lambda cfg: cfg["preprocessing"].update(
                    apply_quality_masking=True, masking_checks=["hands_tracking_loss"]
                ),
                _MASKING_NOT_ENABLED.format(names="['hands_tracking_loss']"),
            ),
        ],
    )
    def test_config_error_exits_1_before_processing(
        self, tmp_path, minimal_config_dict, caplog, edit, message
    ):
        edit(minimal_config_dict)
        minimal_config_dict["input"]["data_dir"] = str(tmp_path / "data")
        minimal_config_dict["output"]["bids_root"] = str(tmp_path / "out")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path)]) == 1
        assert f"Configuration error in {path}: " in caplog.text
        assert message in caplog.text
        assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# pipeline.validate_config — masking checks that never run
# ---------------------------------------------------------------------------


class TestMaskingChecksMustBeEnabled:
    """With masking on, every masking_checks name must also be an enabled check."""

    @staticmethod
    def _config(minimal_config_dict, *, masking: bool, masking_checks) -> PipelineConfig:
        minimal_config_dict["validation"]["enabled_checks"] = ["sampling_rate", "eyes_closed"]
        minimal_config_dict["preprocessing"].update(
            apply_quality_masking=masking, masking_checks=masking_checks
        )
        return PipelineConfig.model_validate(minimal_config_dict)

    def test_masking_on_with_a_check_not_enabled_is_an_error(self, minimal_config_dict):
        config = self._config(
            minimal_config_dict, masking=True, masking_checks=["eyes_closed", "hands_tracking_loss"]
        )
        with pytest.raises(ConfigurationError) as excinfo:
            validate_config(config)
        assert str(excinfo.value) == _MASKING_NOT_ENABLED.format(names="['hands_tracking_loss']")

    def test_masking_off_with_the_same_lists_is_accepted(self, minimal_config_dict):
        config = self._config(
            minimal_config_dict,
            masking=False,
            masking_checks=["eyes_closed", "hands_tracking_loss"],
        )
        validate_config(config)

    @pytest.mark.parametrize("masking_checks", [["eyes_closed", "sampling_rate"], None, []])
    def test_masking_on_with_every_name_enabled_is_accepted(
        self, minimal_config_dict, masking_checks
    ):
        config = self._config(minimal_config_dict, masking=True, masking_checks=masking_checks)
        validate_config(config)

    def test_an_unknown_name_is_reported_once(self, minimal_config_dict):
        config = self._config(
            minimal_config_dict, masking=True, masking_checks=["eye_closed", "stats_summary"]
        )
        with pytest.raises(ConfigurationError) as excinfo:
            validate_config(config)
        message = str(excinfo.value)
        assert "preprocessing.masking_checks names unknown check(s) ['eye_closed']" in message
        assert _MASKING_NOT_ENABLED.format(names="['stats_summary']") in message
        assert message.count("eye_closed'") == 1
