"""Merge native + custom-class events into one wide BIDS events frame, and
build the matching JSON sidecar.

Pure data functions — no file I/O. pipeline.py passes the sidecar dict to the
writer, so this module never imports from io/.
"""

from __future__ import annotations

import pandas as pd

from ..core.exceptions import DataLoadError, ResXRError
from ..core.logger import get_logger
from ..core.session import ColumnInfoEntry, CustomTableSchema

logger = get_logger(__name__)

STANDARD_COLS = ["onset", "duration", "name"]
NA = "n/a"


def merge_events(
    native_events_df: pd.DataFrame | None,
    custom_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge native events and custom-class rows into one wide sparse frame.

    Always returns at least STANDARD_COLS. Missing cells become the string
    "n/a". Rows are sorted by onset ascending.
    """
    # 1. Reserved-name guard: no custom class may carry a 'name' column.
    for cls, df in custom_dfs.items():
        if "name" in df.columns:
            raise ResXRError(
                f"Custom class '{cls}' has a reserved column 'name' which collides "
                f"with the events merge key. Rename it in the Unity export."
            )

    # 2. Native onset guard (NaN passes through to_numeric and would misorder).
    if native_events_df is not None and not native_events_df.empty:
        onset = pd.to_numeric(native_events_df["onset"], errors="coerce")
        if onset.isna().any():
            raise DataLoadError("Native events contain a null/non-numeric onset value.")

    # 3. Cross-class column collision check (outside the standard set).
    exclude = set(STANDARD_COLS)
    seen: dict[str, str] = {}
    for cls, df in custom_dfs.items():
        for col in df.columns:
            if col in exclude:
                continue
            if col in seen:
                raise ResXRError(
                    f"Column '{col}' appears in two custom classes: "
                    f"'{seen[col]}' and '{cls}'. Custom column names must be unique."
                )
            seen[col] = cls

    # 4. Empty case.
    if (native_events_df is None or native_events_df.empty) and not custom_dfs:
        return pd.DataFrame(columns=STANDARD_COLS)

    frames: list[pd.DataFrame] = []

    # 5. Native rows already carry 'name'. Do not re-set it.
    if native_events_df is not None and not native_events_df.empty:
        frames.append(native_events_df.copy())

    # 6. Custom rows get name = class name.
    for cls, df in custom_dfs.items():
        part = df.copy()
        part["name"] = cls
        frames.append(part)

    # 7. Build the ordered column union (standard first, then custom columns in
    #    first-seen order), concat, sort while onset is still numeric, stringify gaps.
    union_cols = list(STANDARD_COLS)
    for f in frames:
        for c in f.columns:
            if c not in union_cols:
                union_cols.append(c)

    reindexed = [f.reindex(columns=union_cols) for f in frames]
    merged = pd.concat(reindexed, ignore_index=True)
    merged = merged.sort_values("onset", kind="stable").reset_index(drop=True)
    merged = merged.fillna(NA)
    return merged


def generate_events_sidecar(
    merged_df: pd.DataFrame,
    custom_tables: list[CustomTableSchema] | None,
) -> dict:
    """Build the events.json sidecar dict from the merged frame + schemas.

    Standard columns are always described. Custom columns are described only
    when a matching ColumnInfoEntry exists; otherwise the column is left out of
    the sidecar (it still appears in the TSV).
    """
    sidecar: dict = {
        "onset": {
            "Description": "Onset time of event in seconds relative to recording start",
            "Units": "s",
        },
        "duration": {
            "Description": "Duration of event in seconds (0 for instantaneous events)",
            "Units": "s",
        },
        "name": {
            "Description": "Event type or custom data-class name",
            "LongName": "Event Type",
            "Levels": {str(v): str(v) for v in merged_df["name"].unique()},
        },
    }

    if not custom_tables:
        return sidecar

    by_name: dict[str, ColumnInfoEntry] = {}
    for table in custom_tables:
        for col in table.columns:
            by_name[col.name] = col

    for col_name in merged_df.columns:
        if col_name in STANDARD_COLS:
            continue
        entry = by_name.get(col_name)
        if entry is None:
            continue
        desc: dict = {"Description": entry.description, "Format": entry.format}
        if entry.units is not None:
            desc["Units"] = entry.units
        if entry.levels is not None:
            desc["Levels"] = entry.levels
        if entry.minimum is not None:
            desc["Minimum"] = entry.minimum
        if entry.maximum is not None:
            desc["Maximum"] = entry.maximum
        sidecar[col_name] = desc

    return sidecar
