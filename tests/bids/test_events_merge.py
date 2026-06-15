"""Tests for events merge + sidecar generation (bids/events_merge.py)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from resxr.bids.events_merge import generate_events_sidecar, merge_events
from resxr.core.exceptions import DataLoadError, ResXRError
from resxr.core.session import ColumnInfoEntry, CustomTableSchema

STANDARD_COLS = ["onset", "duration", "name"]


def _native():
    return pd.DataFrame({"name": ["start", "end"], "onset": [0.0, 10.0], "duration": [0.0, 0.0]})


def _choice():
    return pd.DataFrame(
        {"onset": [5.0], "duration": [1.0], "reaction_time": [0.3], "choice": ["L"]}
    )


class TestMergeEvents:
    def test_native_only(self):
        out = merge_events(_native(), {})
        assert list(out.columns) == STANDARD_COLS
        assert len(out) == 2

    def test_custom_only(self):
        out = merge_events(None, {"ChoiceEvent": _choice()})
        assert set(out.columns) == {"onset", "duration", "name", "reaction_time", "choice"}
        assert list(out["name"]) == ["ChoiceEvent"]

    def test_both_sorted_by_onset(self):
        out = merge_events(_native(), {"ChoiceEvent": _choice()})
        assert list(out["onset"]) == [0.0, 5.0, 10.0]
        assert list(out["name"]) == ["start", "ChoiceEvent", "end"]

    def test_both_absent_returns_empty_standard_frame(self):
        out = merge_events(None, {})
        assert list(out.columns) == STANDARD_COLS
        assert len(out) == 0

    def test_custom_text_columns_not_upcast_by_gap_fill(self):
        # Custom columns arrive as strings (load_custom_class_csv reads dtype=str).
        # When native rows lack these columns, reindex introduces NaN; the merge
        # must fill gaps with "n/a" WITHOUT turning false/true into 0.0/1.0 or
        # 0 into 0.0 on the rows that do have values.
        custom = pd.DataFrame(
            {
                "onset": [5.0, 7.0],
                "duration": [1.0, 1.0],
                "flag": ["false", "true"],
                "trial": ["0", "1"],
            }
        )
        out = merge_events(_native(), {"Maze": custom})
        maze = out[out["name"] == "Maze"]
        assert maze["flag"].tolist() == ["false", "true"]
        assert maze["trial"].tolist() == ["0", "1"]
        # Native rows lack the custom columns -> "n/a", never 0.0/1.0.
        native_rows = out[out["name"].isin(["start", "end"])]
        assert set(native_rows["flag"]) == {"n/a"}
        assert "0.0" not in out["flag"].tolist()
        assert "1.0" not in out["flag"].tolist()

    def test_native_null_onset_raises(self):
        bad = pd.DataFrame({"name": ["x"], "onset": [float("nan")], "duration": [0.0]})
        with pytest.raises(DataLoadError):
            merge_events(bad, {})

    def test_reserved_name_column_in_custom_raises(self):
        bad = pd.DataFrame({"onset": [1.0], "duration": [0.0], "name": ["oops"]})
        with pytest.raises(ResXRError):
            merge_events(None, {"BadClass": bad})

    def test_cross_class_collision_allowed_with_identical_metadata(self):
        a = pd.DataFrame({"onset": [1.0], "duration": [0.0], "shared": [1]})
        b = pd.DataFrame({"onset": [2.0], "duration": [0.0], "shared": [2]})

        # Create identical schemas for the shared column
        col_info_a = ColumnInfoEntry(name="shared", description="Same", format="int")
        col_info_b = ColumnInfoEntry(name="shared", description="Same", format="int")

        schema_a = CustomTableSchema(class_name="ClassA", row_count=1, columns=[col_info_a])
        schema_b = CustomTableSchema(class_name="ClassB", row_count=1, columns=[col_info_b])

        out = merge_events(None, {"ClassA": a, "ClassB": b}, [schema_a, schema_b])
        assert "shared" in out.columns
        assert list(out["shared"]) == [1, 2]

    def test_cross_class_collision_raises_on_metadata_mismatch(self):
        a = pd.DataFrame({"onset": [1.0], "duration": [0.0], "shared": [1]})
        b = pd.DataFrame({"onset": [2.0], "duration": [0.0], "shared": [2]})

        # Create clashing schemas for the shared column (different descriptions)
        col_info_a = ColumnInfoEntry(name="shared", description="Thing A", format="int")
        col_info_b = ColumnInfoEntry(name="shared", description="Thing B", format="int")

        schema_a = CustomTableSchema(class_name="ClassA", row_count=1, columns=[col_info_a])
        schema_b = CustomTableSchema(class_name="ClassB", row_count=1, columns=[col_info_b])

        with pytest.raises(ResXRError) as exc:
            merge_events(None, {"ClassA": a, "ClassB": b}, [schema_a, schema_b])
        assert "conflicting metadata" in str(exc.value)

    def test_cross_class_collision_raises_on_strict_fallback(self):
        a = pd.DataFrame({"onset": [1.0], "duration": [0.0], "shared": [1]})
        b = pd.DataFrame({"onset": [2.0], "duration": [0.0], "shared": [2]})

        with pytest.raises(ResXRError) as exc:
            merge_events(None, {"ClassA": a, "ClassB": b})
        assert "no CustomTables sidecar json was provided" in str(exc.value)

    def test_gaps_are_string_na_not_nan(self):
        out = merge_events(_native(), {"ChoiceEvent": _choice()})
        native_row = out[out["name"] == "start"].iloc[0]
        assert native_row["reaction_time"] == "n/a"
        assert not any(isinstance(v, float) and math.isnan(v) for v in out.to_numpy().ravel())


class TestGenerateEventsSidecar:
    def _tables(self):
        return [
            CustomTableSchema(
                class_name="ChoiceEvent",
                row_count=1,
                columns=[
                    ColumnInfoEntry(
                        name="reaction_time", description="RT", format="float", units="s"
                    ),
                    ColumnInfoEntry(
                        name="choice",
                        description="Choice",
                        format="str",
                        levels={"L": "left", "R": "right"},
                    ),
                ],
            )
        ]

    def test_standard_columns_always_present(self):
        out = merge_events(_native(), {})
        sidecar = generate_events_sidecar(out, None)
        assert "onset" in sidecar and "duration" in sidecar and "name" in sidecar
        assert sidecar["onset"]["Units"] == "s"
        assert "Levels" in sidecar["name"]

    def test_name_levels_is_object(self):
        out = merge_events(_native(), {"ChoiceEvent": _choice()})
        sidecar = generate_events_sidecar(out, self._tables())
        assert isinstance(sidecar["name"]["Levels"], dict)
        assert "ChoiceEvent" in sidecar["name"]["Levels"]

    def test_custom_column_described_from_schema(self):
        out = merge_events(_native(), {"ChoiceEvent": _choice()})
        sidecar = generate_events_sidecar(out, self._tables())
        assert sidecar["reaction_time"]["Description"] == "RT"
        assert sidecar["reaction_time"]["Units"] == "s"
        assert sidecar["reaction_time"]["Format"] == "float"
        assert sidecar["choice"]["Levels"] == {"L": "left", "R": "right"}
        assert "Units" not in sidecar["choice"]

    def test_no_schema_means_no_entry_for_custom_column(self):
        out = merge_events(_native(), {"ChoiceEvent": _choice()})
        sidecar = generate_events_sidecar(out, None)
        assert "reaction_time" not in sidecar
        assert "choice" not in sidecar
        assert "name" in sidecar

    def test_na_token_excluded_from_name_levels(self):
        """The BIDS missing-value token 'n/a' must not appear as a name Level."""
        merged = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "duration": [0.0, 0.0],
                "name": ["real_event", "n/a"],
            }
        )
        sidecar = generate_events_sidecar(merged, None)
        assert "n/a" not in sidecar["name"]["Levels"]
        assert "real_event" in sidecar["name"]["Levels"]

    def test_nan_name_excluded_from_levels(self):
        """A NaN name must not leak into the name Levels as 'nan' or 'n/a'."""
        merged = pd.DataFrame(
            {
                "onset": [0.0, 1.0],
                "duration": [0.0, 0.0],
                "name": ["real_event", float("nan")],
            }
        )
        sidecar = generate_events_sidecar(merged, None)
        assert "nan" not in sidecar["name"]["Levels"]
        assert "n/a" not in sidecar["name"]["Levels"]
        assert "real_event" in sidecar["name"]["Levels"]
