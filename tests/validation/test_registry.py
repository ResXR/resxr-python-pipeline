"""Tests for the CheckRegistry and ValidationCheck protocol (validation/registry.py)."""

from __future__ import annotations

from resxr.core.config import ValidationConfig
from resxr.core.constants import TrackingSystem
from resxr.core.session import QualityFlag
from resxr.validation.registry import CheckRegistry

# ---------------------------------------------------------------------------
# Minimal stub checks for isolated tests
# ---------------------------------------------------------------------------


class _AlwaysEmptyCheck:
    """Check that never produces flags. Has no required_streams (runs on each stream)."""

    name = "always_empty"
    description = "Never flags anything"

    def __call__(self, stream, session, config) -> list[QualityFlag]:
        return []


class _AlwaysFlagCheck:
    """Check that always produces exactly one warning flag."""

    name = "always_flag"
    description = "Always produces one flag"

    def __call__(self, stream, session, config) -> list[QualityFlag]:
        return [
            QualityFlag(
                check_name=self.name,
                system=stream.system,
                start_time=1.0,
                end_time=2.0,
                severity="warning",
                message="Always flag",
                mask=False,
            )
        ]


class _MultiStreamCheck:
    """Multi-stream check that requires HANDS stream (only runs when stream=HANDS)."""

    name = "multi_stream"
    description = "Requires HANDS stream"
    required_streams = [TrackingSystem.HANDS]

    def __call__(self, stream, session, config) -> list[QualityFlag]:
        return [
            QualityFlag(
                check_name=self.name,
                system=TrackingSystem.HANDS,
                start_time=1.0,
                end_time=2.0,
                severity="warning",
                message="Multi-stream flag",
                mask=False,
            )
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validation_config(enabled_checks: list[str], **extra) -> ValidationConfig:
    """Create a ValidationConfig with the given enabled checks and optional extras."""
    settings = dict(extra)
    return ValidationConfig(enabled_checks=enabled_checks, settings=settings)


# ===========================================================================
# CheckRegistry.register
# ===========================================================================


class TestCheckRegistryRegister:
    def test_register_adds_check(self):
        """register() stores check under its name."""
        registry = CheckRegistry()
        check = _AlwaysEmptyCheck()
        registry.register(check)
        assert "always_empty" in registry.available_checks

    def test_register_multiple_checks(self):
        """Multiple checks can be registered."""
        registry = CheckRegistry()
        registry.register(_AlwaysEmptyCheck())
        registry.register(_AlwaysFlagCheck())
        assert "always_empty" in registry.available_checks
        assert "always_flag" in registry.available_checks

    def test_register_overwrites_same_name(self):
        """Re-registering the same name replaces the previous check."""
        registry = CheckRegistry()
        check1 = _AlwaysEmptyCheck()
        check2 = _AlwaysEmptyCheck()  # same name, different object
        registry.register(check1)
        registry.register(check2)
        # Should not raise; second registration wins
        assert registry.get_check("always_empty") is check2

    def test_get_check_returns_registered_check(self):
        """get_check() returns the registered instance."""
        registry = CheckRegistry()
        check = _AlwaysFlagCheck()
        registry.register(check)
        assert registry.get_check("always_flag") is check

    def test_get_check_returns_none_for_unknown(self):
        """get_check() returns None for unregistered names."""
        registry = CheckRegistry()
        assert registry.get_check("nonexistent") is None

    def test_available_checks_is_list_of_strings(self):
        """available_checks returns a list of strings."""
        registry = CheckRegistry()
        registry.register(_AlwaysEmptyCheck())
        assert isinstance(registry.available_checks, list)
        assert all(isinstance(n, str) for n in registry.available_checks)


# ===========================================================================
# CheckRegistry.get_enabled_checks
# ===========================================================================


class TestGetEnabledChecks:
    def test_returns_all_when_all_enabled(self):
        """All registered checks are returned when all their names are listed."""
        registry = CheckRegistry()
        registry.register(_AlwaysEmptyCheck())
        registry.register(_AlwaysFlagCheck())
        result = registry.get_enabled_checks(["always_empty", "always_flag"])
        assert len(result) == 2

    def test_returns_subset_when_some_enabled(self):
        """Only the listed check is returned when a subset is enabled."""
        registry = CheckRegistry()
        registry.register(_AlwaysEmptyCheck())
        registry.register(_AlwaysFlagCheck())
        result = registry.get_enabled_checks(["always_empty"])
        assert len(result) == 1
        assert result[0].name == "always_empty"

    def test_returns_empty_when_none_enabled(self):
        """Empty list is returned when no checks are enabled."""
        registry = CheckRegistry()
        registry.register(_AlwaysEmptyCheck())
        result = registry.get_enabled_checks([])
        assert result == []

    def test_unknown_name_silently_excluded(self):
        """Unregistered check names are silently excluded from the result."""
        registry = CheckRegistry()
        registry.register(_AlwaysEmptyCheck())
        result = registry.get_enabled_checks(["always_empty", "does_not_exist"])
        assert len(result) == 1


# ===========================================================================
# CheckRegistry.run_all
# ===========================================================================


class TestRunAll:
    def test_run_all_returns_list(self, head_stream, minimal_session):
        """run_all returns a list (even if empty)."""
        registry = CheckRegistry()
        config = _make_validation_config([])
        result = registry.run_all(head_stream, minimal_session, config)
        assert isinstance(result, list)

    def test_run_all_no_enabled_checks_returns_empty(self, head_stream, minimal_session):
        """When no checks are enabled, run_all returns empty list."""
        registry = CheckRegistry()
        registry.register(_AlwaysFlagCheck())
        config = _make_validation_config([])  # enabled_checks=[]
        result = registry.run_all(head_stream, minimal_session, config)
        assert result == []

    def test_run_all_aggregates_flags(self, head_stream, minimal_session):
        """run_all combines flags from all enabled checks."""
        registry = CheckRegistry()
        registry.register(_AlwaysFlagCheck())
        config = _make_validation_config(["always_flag"])
        result = registry.run_all(head_stream, minimal_session, config)
        assert len(result) == 1

    def test_run_all_returns_quality_flag_instances(self, head_stream, minimal_session):
        """Every element in the result is a QualityFlag."""
        registry = CheckRegistry()
        registry.register(_AlwaysFlagCheck())
        config = _make_validation_config(["always_flag"])
        result = registry.run_all(head_stream, minimal_session, config)
        for flag in result:
            assert isinstance(flag, QualityFlag)

    def test_run_all_multistream_check_skipped_for_wrong_stream(self, head_stream, full_session):
        """A multi-stream check with required_streams=[HANDS] does NOT run on HEAD stream."""
        registry = CheckRegistry()
        registry.register(_MultiStreamCheck())
        config = _make_validation_config(["multi_stream"])
        result = registry.run_all(head_stream, full_session, config)
        assert result == []

    def test_run_all_multistream_check_runs_for_first_required_stream(
        self, hands_stream, full_session
    ):
        """A multi-stream check runs exactly once when stream==required_streams[0]."""
        registry = CheckRegistry()
        registry.register(_MultiStreamCheck())
        config = _make_validation_config(["multi_stream"])
        result = registry.run_all(hands_stream, full_session, config)
        assert len(result) == 1

    def test_run_all_multistream_check_skipped_when_required_stream_missing(
        self, hands_stream, minimal_session
    ):
        """Multi-stream check is skipped when a required stream is absent from session."""
        # minimal_session has only HEAD — HANDS is missing
        registry = CheckRegistry()
        registry.register(_MultiStreamCheck())
        config = _make_validation_config(["multi_stream"])
        result = registry.run_all(hands_stream, minimal_session, config)
        # HANDS stream itself is there but session.has_stream(HANDS) is False
        # because minimal_session only has HEAD
        assert result == []

    def test_run_all_exception_in_check_does_not_crash(self, head_stream, minimal_session):
        """An exception raised inside a check is caught and does not propagate."""

        class _BrokenCheck:
            name = "broken"
            description = "Always raises"

            def __call__(self, stream, session, config):
                raise RuntimeError("Intentional error")

        registry = CheckRegistry()
        registry.register(_BrokenCheck())
        config = _make_validation_config(["broken"])
        # Should not raise
        result = registry.run_all(head_stream, minimal_session, config)
        assert isinstance(result, list)


# ===========================================================================
# ValidationCheck protocol
# ===========================================================================


class TestValidationCheckProtocol:
    def test_check_has_name(self):
        """A valid check has a non-empty string name."""
        check = _AlwaysEmptyCheck()
        assert isinstance(check.name, str)
        assert len(check.name) > 0

    def test_check_has_description(self):
        """A valid check has a non-empty string description."""
        check = _AlwaysEmptyCheck()
        assert isinstance(check.description, str)
        assert len(check.description) > 0

    def test_check_is_callable(self):
        """A valid check is callable."""
        check = _AlwaysEmptyCheck()
        assert callable(check)

    def test_check_without_required_streams_attribute(self):
        """A check without required_streams is compatible with the registry."""
        check = _AlwaysEmptyCheck()
        # getattr fallback used by registry should return None
        assert getattr(check, "required_streams", None) is None


# ===========================================================================
# register_check decorator
# ===========================================================================


class TestRegisterCheckDecorator:
    def test_decorator_registers_to_global_registry(self):
        """register_check(instance) adds the check to the global check_registry."""
        from resxr.validation.registry import check_registry, register_check

        class _UniqueDecoratorCheck:
            name = "_test_decorator_unique_12345"
            description = "Decorator test"

            def __call__(self, stream, session, config):
                return []

        instance = _UniqueDecoratorCheck()
        returned = register_check(instance)
        # Returns the check unchanged (passthrough decorator pattern)
        assert returned is instance
        # Check is now in the global registry
        assert check_registry.get_check("_test_decorator_unique_12345") is instance

    def test_real_checks_registered_at_import_time(self):
        """Built-in checks (sampling_rate, stats_summary) are in the global registry."""
        # Force import of all check modules so they register
        import resxr.validation.checks.sampling_rate  # noqa: F401
        import resxr.validation.checks.stats  # noqa: F401
        from resxr.validation.registry import check_registry

        assert "sampling_rate" in check_registry.available_checks
        assert "stats_summary" in check_registry.available_checks


# ===========================================================================
# run_all does NOT attach flags to streams
# ===========================================================================


class TestRunAllDoesNotAttachFlags:
    def test_stream_quality_flags_unchanged_after_run_all(self, head_stream, minimal_session):
        """run_all returns flags but does NOT mutate stream.quality_flags."""
        registry = CheckRegistry()
        registry.register(_AlwaysFlagCheck())
        config = _make_validation_config(["always_flag"])

        original_flags = list(head_stream.quality_flags)
        returned_flags = registry.run_all(head_stream, minimal_session, config)

        # Flags were returned
        assert len(returned_flags) == 1
        # But stream.quality_flags was NOT modified
        assert head_stream.quality_flags == original_flags
