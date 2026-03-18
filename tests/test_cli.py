"""Tests for the ResXR command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from resxr.cli import build_parser, main


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
