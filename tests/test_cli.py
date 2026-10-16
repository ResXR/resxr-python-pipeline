"""Tests for the ResXR command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from resxr.cli import build_parser, main
from resxr.core.config import PipelineConfig
from resxr.core.exceptions import ConfigurationError
from resxr.pipeline import load_config, validate_config

_MASKING_NOT_ENABLED = (
    "preprocessing.masking_checks names check(s) {names} that are not in "
    "validation.enabled_checks; they never run, so nothing would be masked for them. "
    "Add them to enabled_checks or remove them from masking_checks."
)

# Every validation.settings key read in src/ (the shipped checks and get_column_groups)
_SHIPPED_SETTINGS_KEYS = [
    "check_column_groups",
    "eyes_closed_min_duration",
    "eyes_closed_threshold",
    "eyes_closed_use_min_duration",
    "sampling_cv_threshold",
    "sampling_rate_tolerance",
    "tracking_flags",
]
_UNKNOWN_SETTINGS = (
    "validation.settings has unknown key(s) {keys}; keys the registered checks read: {known}"
)


def _mapping(source_dir: str, subject_id: str, session_label: str = "01") -> dict:
    return {"source_dir": source_dir, "subject_id": subject_id, "session_label": session_label}


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
        minimal_config_dict["input"]["data_dir"] = str(tmp_path)
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
        assert f"Configuration error in {path}: " in caplog.text
        assert _MASKING_NOT_ENABLED.format(names="['eyes_closed']") in caplog.text

    def test_dry_run_rejects_an_unknown_nested_key(self, tmp_path, minimal_config_dict, caplog):
        minimal_config_dict["preprocessing"]["apply_quality_maskin"] = True
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert f"Configuration error in {path}: " in caplog.text
        assert "preprocessing.apply_quality_maskin\n  Extra inputs are not permitted" in (
            caplog.text
        )

    def test_dry_run_rejects_an_unknown_settings_key(self, tmp_path, minimal_config_dict, caplog):
        minimal_config_dict["validation"]["settings"] = {"sampling_rate_tolerence": 0.2}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert f"Configuration error in {path}: " in caplog.text
        assert (
            _UNKNOWN_SETTINGS.format(
                keys="['sampling_rate_tolerence']", known=_SHIPPED_SETTINGS_KEYS
            )
            in caplog.text
        )

    def test_dry_run_rejects_a_missing_rate_for_an_enabled_system(
        self, tmp_path, minimal_config_dict, caplog
    ):
        del minimal_config_dict["sampling_frequencies"]["Eyes"]
        del minimal_config_dict["sampling_frequencies"]["Body"]
        minimal_config_dict["systems"] = {"Body": {"enabled": False}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert "sampling_frequencies has no entry for ['Eyes'], which is enabled;" in caplog.text
        assert "Body" not in caplog.text  # disabled, so it needs no rate

    def test_dry_run_rejects_an_unknown_system_in_system_descriptions(
        self, tmp_path, minimal_config_dict, caplog
    ):
        minimal_config_dict["system_descriptions"]["Hand"] = "Hand tracking"
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert (
            f"Configuration error in {path}: system_descriptions names unknown tracking "
            "system(s) ['Hand']; valid names: ['Head', 'Hands', 'Eyes', 'Face', 'Body', "
            "'Controllers']"
        ) in caplog.text

    def test_dry_run_rejects_a_data_dir_that_does_not_exist(
        self, tmp_path, minimal_config_dict, caplog
    ):
        minimal_config_dict["input"]["data_dir"] = str(tmp_path / "no_such_folder")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path), "--dry-run"]) == 1
        assert (
            f"Configuration error in {path}: input.data_dir does not exist: "
            f"{tmp_path / 'no_such_folder'}"
        ) in caplog.text
        assert "Configuration valid" not in caplog.text

    def test_a_relative_data_dir_is_named_with_the_folder_it_is_read_from(
        self, tmp_path, minimal_config_dict, caplog, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a_file").write_text("x")
        for data_dir, state in (("missing", "does not exist"), ("a_file", "is not a folder")):
            minimal_config_dict["input"]["data_dir"] = data_dir
            path = tmp_path / "config.yaml"
            path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
            assert main(["--config", str(path), "--dry-run"]) == 1
            assert (
                f"input.data_dir {state}: {data_dir} ({tmp_path / data_dir}; a relative path "
                "is read from the current folder)"
            ) in caplog.text


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
                lambda cfg: cfg["system_descriptions"].update(Hand="Hand tracking"),
                "system_descriptions names unknown tracking system(s) ['Hand']; valid names: ",
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
            (
                lambda cfg: cfg["preprocessing"].update(apply_quality_maskin=True),
                "preprocessing.apply_quality_maskin\n  Extra inputs are not permitted",
            ),
            (
                lambda cfg: cfg["validation"].update(settings={"eyes_closed_treshold": 0.8}),
                _UNKNOWN_SETTINGS.format(
                    keys="['eyes_closed_treshold']", known=_SHIPPED_SETTINGS_KEYS
                ),
            ),
            (
                lambda cfg: cfg.update(outputs={"bids_root": "elsewhere"}),
                "Unknown top-level key(s) ['outputs']; check for typos. Known top-level keys: ",
            ),
            (
                lambda cfg: cfg.update(session_mappings=[_mapping("a", "01"), _mapping("b", "01")]),
                "session_mappings maps 2 folders to sub-01 ses-01: a, b; each subject/session "
                "pair must be unique",
            ),
            (
                lambda cfg: cfg.update(session_mappings=[_mapping("a", "01"), _mapping("a", "02")]),
                " 2 times (sub-01 ses-01, sub-02 ses-01); map each folder once",
            ),
            (
                lambda cfg: cfg.update(session_mappings=[_mapping("a", "sub-01")]),
                "BIDS labels may hold only letters and digits (the sub-, ses- and task- prefixes "
                "are added): subject_id of a 'sub-01'",
            ),
            (
                lambda cfg: cfg["sampling_frequencies"].pop("Eyes"),
                "sampling_frequencies has no entry for ['Eyes'], which is enabled; add the "
                "expected rate in Hz, or set systems.<name>.enabled: false for a system that was "
                "not recorded",
            ),
            (
                lambda cfg: cfg["output"].update(task_name="VR tracking"),
                "BIDS labels may hold only letters and digits (the sub-, ses- and task- prefixes "
                "are added): output.task_name 'VR tracking'",
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


# ---------------------------------------------------------------------------
# pipeline.validate_config — validation.settings keys must be declared
# ---------------------------------------------------------------------------


class TestSettingsKeysMustBeDeclared:
    """Every validation.settings key must be declared by a registered check."""

    @staticmethod
    def _config(minimal_config_dict, settings: dict) -> PipelineConfig:
        minimal_config_dict["validation"]["settings"] = settings
        return PipelineConfig.model_validate(minimal_config_dict)

    def test_a_misspelled_key_is_an_error_naming_the_known_keys(self, minimal_config_dict):
        config = self._config(
            minimal_config_dict, {"eyes_closed_threshold": 0.8, "trackin_flags": {}}
        )
        with pytest.raises(ConfigurationError) as excinfo:
            validate_config(config)
        assert str(excinfo.value) == _UNKNOWN_SETTINGS.format(
            keys="['trackin_flags']", known=_SHIPPED_SETTINGS_KEYS
        )

    def test_every_key_the_shipped_code_reads_is_accepted(self, minimal_config_dict):
        validate_config(self._config(minimal_config_dict, dict.fromkeys(_SHIPPED_SETTINGS_KEYS)))

    def test_the_shipped_config_passes_validate_config(self, monkeypatch):
        root = Path(__file__).resolve().parents[1]
        monkeypatch.chdir(root)  # input.data_dir (DATA) is read from the repository root
        load_config(root / "config" / "pipeline_config.yaml")

    @pytest.mark.parametrize("declared", [("my_check_threshold",), "my_check_threshold"])
    def test_a_custom_check_declares_its_own_key(self, minimal_config_dict, monkeypatch, declared):
        from resxr.validation import check_registry

        class _MyCheck:
            name = "_my_custom_check"
            description = "Reads its own settings key"
            required_streams = None
            settings_keys = declared

            def __call__(self, stream, session, config):
                return []

        config = self._config(minimal_config_dict, {"my_check_threshold": 0.5})
        with pytest.raises(ConfigurationError, match=r"unknown key\(s\) \['my_check_threshold'\]"):
            validate_config(config)
        monkeypatch.setitem(check_registry._checks, _MyCheck.name, _MyCheck())
        validate_config(config)


# ---------------------------------------------------------------------------
# pipeline.validate_config — session mappings and BIDS labels
# ---------------------------------------------------------------------------


class TestSessionMappingsAndLabels:
    """Mappings must be unique and labels must be BIDS labels (letters and digits)."""

    @staticmethod
    def _error(minimal_config_dict, mappings, task_name="vr") -> str:
        minimal_config_dict["session_mappings"] = mappings
        minimal_config_dict["output"]["task_name"] = task_name
        with pytest.raises(ConfigurationError) as excinfo:
            validate_config(PipelineConfig.model_validate(minimal_config_dict))
        return str(excinfo.value)

    def test_three_folders_mapped_to_one_pair_are_all_named(self, minimal_config_dict):
        mappings = [
            _mapping("a", "01"),
            _mapping("b", "02"),
            _mapping("c", "01"),
            _mapping("d", "01"),
        ]
        assert self._error(minimal_config_dict, mappings) == (
            "session_mappings maps 3 folders to sub-01 ses-01: a, c, d; each subject/session "
            "pair must be unique"
        )

    def test_one_folder_mapped_twice_is_named_with_both_targets(
        self, minimal_config_dict, tmp_path
    ):
        minimal_config_dict["input"]["data_dir"] = str(tmp_path)
        mappings = [_mapping("a", "01"), _mapping(str(tmp_path / "a"), "02", "02")]
        assert self._error(minimal_config_dict, mappings) == (
            f"session_mappings maps the folder {(tmp_path / 'a').resolve()} 2 times "
            "(sub-01 ses-01, sub-02 ses-02); map each folder once"
        )

    def test_every_invalid_label_is_named_with_its_field(self, minimal_config_dict):
        mappings = [_mapping("a", "P-01"), _mapping("b", "02", "1_a"), _mapping("c", "")]
        assert self._error(minimal_config_dict, mappings, task_name="task-vr") == (
            "BIDS labels may hold only letters and digits (the sub-, ses- and task- prefixes are "
            "added): output.task_name 'task-vr', subject_id of a 'P-01', session_label of b "
            "'1_a', subject_id of c ''"
        )

    @pytest.mark.parametrize("label", ["../evil", "a b", "é1", "01\n"])
    def test_separators_spaces_and_non_ascii_are_not_labels(self, minimal_config_dict, label):
        message = self._error(minimal_config_dict, [_mapping("a", label)])
        assert f"subject_id of a {label!r}" in message

    def test_unique_mappings_with_letter_and_digit_labels_are_accepted(self, minimal_config_dict):
        minimal_config_dict["session_mappings"] = [
            _mapping("a", "01"),
            _mapping("b", "01", "02"),
            _mapping("c", "P001", "pre"),
        ]
        validate_config(PipelineConfig.model_validate(minimal_config_dict))


class TestEnabledSystemsNeedARate:
    """Every tracking system not disabled under systems needs a sampling rate."""

    def test_two_missing_rates_are_named_together(self, minimal_config_dict):
        del minimal_config_dict["sampling_frequencies"]["Face"]
        del minimal_config_dict["sampling_frequencies"]["Controllers"]
        with pytest.raises(ConfigurationError) as excinfo:
            validate_config(PipelineConfig.model_validate(minimal_config_dict))
        assert str(excinfo.value).startswith(
            "sampling_frequencies has no entry for ['Face', 'Controllers'], which are enabled;"
        )

    def test_a_disabled_system_needs_no_rate(self, minimal_config_dict):
        del minimal_config_dict["sampling_frequencies"]["Controllers"]
        minimal_config_dict["systems"] = {"Controllers": {"enabled": False}}
        validate_config(PipelineConfig.model_validate(minimal_config_dict))


class TestDataDirMustExist:
    """input.data_dir is checked at start-up, with mappings and with auto-discovery."""

    @pytest.mark.parametrize("mappings", [[_mapping("sess_a", "01")], []])
    def test_a_missing_data_dir_stops_the_run_before_any_session(
        self, tmp_path, minimal_config_dict, caplog, mappings
    ):
        minimal_config_dict["input"]["data_dir"] = str(tmp_path / "no_such_folder")
        minimal_config_dict["output"]["bids_root"] = str(tmp_path / "out")
        minimal_config_dict["session_mappings"] = mappings
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(minimal_config_dict), encoding="utf-8")
        assert main(["--config", str(path)]) == 1
        assert (
            f"Configuration error in {path}: input.data_dir does not exist: "
            f"{tmp_path / 'no_such_folder'}"
        ) in caplog.text
        assert "Pipeline failed" not in caplog.text
        assert "Session directory not found" not in caplog.text
        assert not (tmp_path / "out").exists()
