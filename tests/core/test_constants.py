"""Tests for constants and enumerations (core/constants.py)."""

from __future__ import annotations

import pytest

from resxr.core.constants import (
    BIDS_CHANNEL_PATTERNS,
    BIDS_CHANNEL_TYPE_COUNTS,
    COLUMN_SUFFIXES,
    SYSTEM_COLUMN_PREFIXES,
    TrackingSystem,
)


class TestTrackingSystem:
    def test_all_expected_members_present(self):
        """TrackingSystem has exactly the six expected members."""
        expected = {"HEAD", "HANDS", "EYES", "FACE", "BODY", "CONTROLLERS"}
        actual = {m.name for m in TrackingSystem}
        assert actual == expected

    def test_member_count(self):
        assert len(TrackingSystem) == 6

    def test_values_are_strings(self):
        """All enum values are non-empty strings."""
        for member in TrackingSystem:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    @pytest.mark.parametrize(
        "name,value",
        [
            ("HEAD", "Head"),
            ("HANDS", "Hands"),
            ("EYES", "Eyes"),
            ("FACE", "Face"),
            ("BODY", "Body"),
            ("CONTROLLERS", "Controllers"),
        ],
    )
    def test_member_values(self, name, value):
        """Each member has the expected string value."""
        assert TrackingSystem[name].value == value


class TestSystemColumnPrefixes:
    def test_all_tracking_systems_have_prefixes(self):
        """Every TrackingSystem member is a key in SYSTEM_COLUMN_PREFIXES."""
        for system in TrackingSystem:
            assert system in SYSTEM_COLUMN_PREFIXES, f"{system} missing from SYSTEM_COLUMN_PREFIXES"

    def test_each_system_has_nonempty_prefix_list(self):
        """Each system's prefix list is a non-empty list of strings."""
        for system, prefixes in SYSTEM_COLUMN_PREFIXES.items():
            assert isinstance(prefixes, list), f"{system}: expected list"
            assert len(prefixes) > 0, f"{system}: prefix list is empty"
            for p in prefixes:
                assert isinstance(p, str) and len(p) > 0, f"{system}: invalid prefix {p!r}"

    def test_no_duplicate_prefixes_across_systems(self):
        """No prefix string appears in more than one system's list (prevents ambiguity)."""
        all_prefixes: list[str] = []
        for prefixes in SYSTEM_COLUMN_PREFIXES.values():
            all_prefixes.extend(prefixes)
        assert len(all_prefixes) == len(set(all_prefixes)), "Duplicate prefix found across systems"


class TestBidsChannelPatterns:
    def test_nonempty(self):
        """BIDS_CHANNEL_PATTERNS has at least one entry."""
        assert len(BIDS_CHANNEL_PATTERNS) > 0

    def test_each_entry_is_4_tuple_of_strings(self):
        """Every entry is a (suffix, type, component, units) 4-tuple of strings."""
        for i, entry in enumerate(BIDS_CHANNEL_PATTERNS):
            assert len(entry) == 4, f"Entry {i} has {len(entry)} elements, expected 4"
            for j, val in enumerate(entry):
                assert isinstance(val, str), f"Entry {i}[{j}] = {val!r} is not a string"

    def test_known_suffix_present(self):
        """The '_px' suffix (position X) is present in the patterns."""
        suffixes = [pat[0] for pat in BIDS_CHANNEL_PATTERNS]
        assert "_px" in suffixes

    def test_latency_suffix_present(self):
        """The '_Time' LATENCY suffix is present."""
        suffixes = [pat[0] for pat in BIDS_CHANNEL_PATTERNS]
        assert "_Time" in suffixes


class TestColumnSuffixes:
    def test_is_nonempty_list(self):
        """COLUMN_SUFFIXES is a non-empty list."""
        assert isinstance(COLUMN_SUFFIXES, list)
        assert len(COLUMN_SUFFIXES) > 0

    def test_all_elements_are_nonempty_strings(self):
        """Every element is a non-empty string."""
        for s in COLUMN_SUFFIXES:
            assert isinstance(s, str) and len(s) > 0


class TestBidsChannelTypeCounts:
    def test_all_values_end_in_channel_count(self):
        """All count key strings end with 'ChannelCount'."""
        for key, value in BIDS_CHANNEL_TYPE_COUNTS.items():
            assert value.endswith("ChannelCount"), f"{key}: {value!r}"

    def test_expected_types_present(self):
        """Key types like POS, ORNT, LATENCY, MISC are present."""
        for expected in ("POS", "ORNT", "LATENCY", "MISC"):
            assert expected in BIDS_CHANNEL_TYPE_COUNTS
