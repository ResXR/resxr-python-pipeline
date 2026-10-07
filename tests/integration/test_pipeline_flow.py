"""Integration tests for full pipeline orchestration."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from resxr.pipeline import run

_REAL_CSV_METADATA = {
    "session_id": "real_csv_session",
    "utc_start_iso8601": "2024-01-01T12:00:00",
    "device_utc_offset": "+00:00",
    "unity_version": "2022.3.0f1",
    "platform": "Android",
    "build_id": "test_build",
    "ovrplugin_runtime_version": "60.0.0",
    "sampling_mode": "fixed",
    "fixedDeltaTime": 0.011111,
    "schema_rev": "2.9",
    "face_enabled": False,
    "body_enabled": True,
    "hands_enabled": True,
    "eyes_enabled": True,
    "controllers_enabled": True,
    "schema_hand_bones": 24,
    "schema_body_joints": 70,
}

_REPO_REAL_CSV = (
    Path(__file__).resolve().parents[2]
    / "DATA"
    / "test_data"
    / "2026.03.12_15-47_ContinuousData.csv"
)


# ── Helpers ──────────────────────────────────────────────────────────


def _write_session_dir(
    root: Path,
    session_name: str,
    *,
    session_id: str,
    with_events: bool = True,
) -> Path:
    """Create a minimal valid session directory on disk."""
    session_dir = root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    continuous = pd.DataFrame(
        {
            "timeSinceStartup": [1.0, 1.011111, 1.022222],
            "Node_Head_px": [0.1, 0.11, 0.12],
            "Node_Head_py": [1.5, 1.51, 1.49],
            "Node_Head_pz": [0.0, 0.01, -0.01],
            "LeftHand_Root_px": [0.3, 0.31, 0.29],
            "RightHand_Root_px": [-0.3, -0.31, -0.29],
            "LeftHand_Status_HandTracked": [1, 1, 1],
            "RightHand_Status_HandTracked": [1, 1, 1],
        }
    )
    continuous.to_csv(session_dir / f"{session_id}_ContinuousData.csv", index=False)

    metadata = {
        "session_id": session_id,
        "utc_start_iso8601": "2024-01-01T12:00:00",
        "device_utc_offset": "+00:00",
        "unity_version": "2022.3.0f1",
        "platform": "Android",
        "build_id": "test_build",
        "ovrplugin_runtime_version": "60.0.0",
        "sampling_mode": "fixed",
        "fixedDeltaTime": 0.011111,
        "schema_rev": "2.9",
        "face_enabled": False,
        "body_enabled": False,
        "hands_enabled": True,
        "eyes_enabled": False,
        "controllers_enabled": False,
        "schema_hand_bones": 24,
        "schema_body_joints": 0,
    }
    with open(session_dir / "session_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    if with_events:
        events = pd.DataFrame(
            {
                "name": ["start", "stimulus_1", "end"],
                "onset": [0.0, 0.5, 1.0],
                "duration": [0.0, 0.2, 0.0],
            }
        )
        events.to_csv(session_dir / f"{session_id}_EventsData.csv", index=False)

    return session_dir


def _extract_system_from_motion_filename(path: Path) -> str:
    """Return tracking system token from '*_tracksys-<System>_motion.tsv'."""
    marker = "_tracksys-"
    name = path.name
    start = name.index(marker) + len(marker)
    end = name.index("_motion.tsv")
    return name[start:end]


def _write_config(
    tmp_path: Path,
    minimal_config_dict: dict,
    *,
    data_dir: Path,
    bids_root: Path,
    session_mappings: list[dict],
) -> Path:
    cfg = dict(minimal_config_dict)
    cfg["input"] = dict(cfg["input"])
    cfg["output"] = dict(cfg["output"])
    cfg["report"] = dict(cfg["report"])

    cfg["input"]["data_dir"] = str(data_dir)
    cfg["output"]["bids_root"] = str(bids_root)
    cfg["report"]["enabled"] = False
    cfg["session_mappings"] = session_mappings

    config_path = tmp_path / "pipeline_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return config_path


# ── Shared fixture for real-CSV pipeline run ─────────────────────────


@pytest.fixture()
def real_csv_pipeline_output(tmp_path, minimal_config_dict):
    """Run the pipeline on the repo's real CSV and return output paths.

    Skips the test automatically when the real CSV is not on disk.
    """
    if not _REPO_REAL_CSV.exists():
        pytest.skip(f"Real CSV not found: {_REPO_REAL_CSV}")

    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    session_dir = data_dir / "sess_real"
    session_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(_REPO_REAL_CSV, session_dir / "real_ContinuousData.csv")

    with open(session_dir / "session_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_REAL_CSV_METADATA, f)

    cfg_path = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_real", "subject_id": "01", "session_label": "01"},
        ],
    )

    run(str(cfg_path))

    raw_motion_dir = bids_root / "sub-01" / "ses-01" / "motion"
    deriv_motion_dir = bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01" / "motion"

    return SimpleNamespace(
        real_csv=_REPO_REAL_CSV,
        bids_root=bids_root,
        raw_motion_dir=raw_motion_dir,
        deriv_motion_dir=deriv_motion_dir,
        source_df=pd.read_csv(_REPO_REAL_CSV),
    )


# ── Synthetic-data integration tests ────────────────────────────────


def test_pipeline_multi_session_happy_path_writes_outputs(tmp_path, minimal_config_dict):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    _write_session_dir(data_dir, "sess_a", session_id="A")
    _write_session_dir(data_dir, "sess_b", session_id="B")

    cfg_path = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
            {"source_dir": "sess_b", "subject_id": "02", "session_label": "01"},
        ],
    )

    run(str(cfg_path))

    assert (bids_root / "participants.tsv").exists()
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    assert set(participants["participant_id"]) == {"sub-01", "sub-02"}

    for subject in ("01", "02"):
        raw_session_dir = bids_root / f"sub-{subject}" / "ses-01"
        deriv_session_dir = bids_root / "derivatives" / "resxr" / f"sub-{subject}" / "ses-01"
        raw_motion_dir = bids_root / f"sub-{subject}" / "ses-01" / "motion"
        deriv_motion_dir = (
            bids_root / "derivatives" / "resxr" / f"sub-{subject}" / "ses-01" / "motion"
        )
        assert list(raw_motion_dir.glob("*_motion.tsv"))
        assert list(raw_motion_dir.glob("*_channels.tsv"))
        assert list(raw_session_dir.glob("*_scans.tsv"))
        assert list(deriv_motion_dir.glob("*_motion.tsv"))
        assert list(deriv_motion_dir.glob("*_channels.tsv"))
        assert list(deriv_session_dir.glob("*_scans.tsv"))


def test_pipeline_multi_session_partial_failure_continues(tmp_path, minimal_config_dict):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    _write_session_dir(data_dir, "sess_good", session_id="GOOD")

    cfg_path = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_good", "subject_id": "01", "session_label": "01"},
            {"source_dir": "sess_missing", "subject_id": "02", "session_label": "01"},
        ],
    )

    run(str(cfg_path))

    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    assert list(participants["participant_id"]) == ["sub-01"]

    assert (bids_root / "sub-01" / "ses-01" / "motion").exists()
    assert not (bids_root / "sub-02").exists()


def _edit_continuous(session_dir: Path, edit) -> None:
    (path,) = session_dir.glob("*_ContinuousData.csv")
    edit(pd.read_csv(path)).to_csv(path, index=False)


def _zero_hands_clock(session_dir: Path) -> None:
    _edit_continuous(session_dir, lambda df: df.assign(Node_HandLeft_Time=0.0))


def _zero_main_clock(session_dir: Path) -> None:
    _edit_continuous(session_dir, lambda df: df.assign(timeSinceStartup=0.0))


def _keep_one_row(session_dir: Path) -> None:
    _edit_continuous(session_dir, lambda df: df.iloc[:1])


def _add_face_file_without_clocks(session_dir: Path) -> None:
    metadata_path = session_dir / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["face_enabled"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    face = pd.DataFrame({"Eyes_Closed_L": [0.0, 0.0, 0.0]})
    face.to_csv(session_dir / "BAD_FaceExpressionData.csv", index=False)


@pytest.mark.parametrize(
    ("make_untimeable", "message"),
    [
        (_zero_hands_clock, "Stream Hands: no non-zero timestamp found"),
        (_zero_main_clock, "Stream Head: no non-zero timestamp found"),
        (_keep_one_row, "Stream Head: need at least 2 rows"),
        (_add_face_file_without_clocks, "Stream Face: missing 'timestamp' column"),
    ],
)
def test_untimeable_session_is_skipped_and_the_run_continues(
    tmp_path, minimal_config_dict, caplog, make_untimeable, message
):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    for name, session_id in (("sess_bad", "BAD"), ("sess_good", "GOOD")):
        session_dir = _write_session_dir(data_dir, name, session_id=session_id)
        _edit_continuous(
            session_dir, lambda df: df.assign(Node_HandLeft_Time=df["timeSinceStartup"] + 100.0)
        )
    make_untimeable(data_dir / "sess_bad")
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Hands": "Node_HandLeft_Time",
        "Face": "Face_Time",
    }
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_bad", "subject_id": "01", "session_label": "01"},
            {"source_dir": "sess_good", "subject_id": "02", "session_label": "01"},
        ],
    )
    run(str(cfg))
    assert f"Failed to process sess_bad: {message}" in caplog.text
    assert not (bids_root / "sub-01").exists()
    assert list((bids_root / "sub-02" / "ses-01" / "motion").glob("*_motion.tsv"))
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    assert list(participants["participant_id"]) == ["sub-02"]
    assert (bids_root / "dataset_description.json").exists()
    assert (bids_root / "README").exists()


def test_motion_tsv_columns_match_channels_contract(tmp_path, minimal_config_dict):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    _write_session_dir(data_dir, "sess_a", session_id="A", with_events=True)

    cfg_path = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
        ],
    )

    run(str(cfg_path))

    motion_dir = bids_root / "sub-01" / "ses-01" / "motion"
    head_channels = next(motion_dir.glob("*tracksys-Head_channels.tsv"))
    head_motion = next(motion_dir.glob("*tracksys-Head_motion.tsv"))

    channels_df = pd.read_csv(head_channels, sep="\t")
    channel_names = list(channels_df["name"])

    motion_df = pd.read_csv(head_motion, sep="\t", header=None, names=channel_names)

    assert list(motion_df.columns) == channel_names
    assert "timestamp" not in channel_names
    assert "timeSinceStartup" not in channel_names
    assert "latency" in channel_names
    assert "Node_Head_px" in channel_names

    events_file = bids_root / "sub-01" / "ses-01" / "sub-01_ses-01_task-vr_events.tsv"
    assert events_file.exists()
    events_df = pd.read_csv(events_file, sep="\t")
    assert list(events_df.columns[:3]) == ["onset", "duration", "name"]

    deriv_motion_dir = bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01" / "motion"
    assert not list(deriv_motion_dir.glob("*_events.tsv"))


def test_motion_tsv_missing_values_use_config_token(tmp_path, minimal_config_dict):
    """NaN values are serialized as the configured missing token in motion TSV."""
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    session_dir = data_dir / "sess_nan"
    session_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "timeSinceStartup": [1.0, 1.011111, 1.022222],
            "Node_Head_px": [0.1, float("nan"), 0.12],
            "Node_Head_py": [1.5, 1.51, 1.49],
            "Node_Head_pz": [0.0, 0.01, -0.01],
        }
    )
    df.to_csv(session_dir / "nan_ContinuousData.csv", index=False)

    with open(session_dir / "session_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": "nan_session",
                "utc_start_iso8601": "2024-01-01T12:00:00",
                "device_utc_offset": "+00:00",
                "unity_version": "2022.3.0f1",
                "platform": "Android",
                "build_id": "test_build",
                "ovrplugin_runtime_version": "60.0.0",
                "sampling_mode": "fixed",
                "fixedDeltaTime": 0.011111,
                "schema_rev": "2.9",
                "face_enabled": False,
                "body_enabled": False,
                "hands_enabled": False,
                "eyes_enabled": False,
                "controllers_enabled": False,
                "schema_hand_bones": 0,
                "schema_body_joints": 0,
            },
            f,
        )

    cfg = dict(minimal_config_dict)
    cfg["input"] = dict(cfg["input"])
    cfg["output"] = dict(cfg["output"])
    cfg["report"] = dict(cfg["report"])
    cfg["bids"] = dict(cfg["bids"])
    cfg["input"]["data_dir"] = str(data_dir)
    cfg["output"]["bids_root"] = str(bids_root)
    cfg["report"]["enabled"] = False
    cfg["bids"]["missing_values"] = "n/a"
    cfg["session_mappings"] = [
        {"source_dir": "sess_nan", "subject_id": "01", "session_label": "01"},
    ]
    cfg_path = tmp_path / "pipeline_config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    run(str(cfg_path))

    motion_path = next(
        (bids_root / "sub-01" / "ses-01" / "motion").glob("*tracksys-Head_motion.tsv")
    )
    content = motion_path.read_text(encoding="utf-8")
    assert "n/a" in content


def test_pipeline_output_column_order_stable_across_runs(tmp_path, minimal_config_dict):
    """Same input/config produces stable channels order and identical motion text."""
    data_dir = tmp_path / "sessions"
    _write_session_dir(data_dir, "sess_order", session_id="ORDER", with_events=False)

    bids_root_a = tmp_path / "bids_out_a"
    bids_root_b = tmp_path / "bids_out_b"

    cfg_a = _write_config(
        tmp_path / "run_a",
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root_a,
        session_mappings=[{"source_dir": "sess_order", "subject_id": "01", "session_label": "01"}],
    )
    cfg_b = _write_config(
        tmp_path / "run_b",
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root_b,
        session_mappings=[{"source_dir": "sess_order", "subject_id": "01", "session_label": "01"}],
    )

    run(str(cfg_a))
    run(str(cfg_b))

    motion_dir_a = bids_root_a / "sub-01" / "ses-01" / "motion"
    motion_dir_b = bids_root_b / "sub-01" / "ses-01" / "motion"

    channels_a = sorted(motion_dir_a.glob("*_channels.tsv"))
    channels_b = sorted(motion_dir_b.glob("*_channels.tsv"))
    motions_a = sorted(motion_dir_a.glob("*_motion.tsv"))
    motions_b = sorted(motion_dir_b.glob("*_motion.tsv"))

    assert [p.name for p in channels_a] == [p.name for p in channels_b]
    assert [p.name for p in motions_a] == [p.name for p in motions_b]

    for a_path, b_path in zip(channels_a, channels_b, strict=True):
        names_a = list(pd.read_csv(a_path, sep="\t")["name"])
        names_b = list(pd.read_csv(b_path, sep="\t")["name"])
        assert names_a == names_b

    for a_path, b_path in zip(motions_a, motions_b, strict=True):
        assert a_path.read_text(encoding="utf-8") == b_path.read_text(encoding="utf-8")


# ── Real-CSV integration tests (all share the fixture) ──────────────


def test_pipeline_real_csv_full_column_contract(real_csv_pipeline_output):
    """Each system's channels.tsv/motion.tsv pair is internally consistent."""
    out = real_csv_pipeline_output
    motion_files = sorted(out.raw_motion_dir.glob("*_motion.tsv"))
    channels_files = sorted(out.raw_motion_dir.glob("*_channels.tsv"))

    assert len(motion_files) >= 3
    assert len(motion_files) == len(channels_files)

    for motion_path in motion_files:
        channels_path = out.raw_motion_dir / motion_path.name.replace(
            "_motion.tsv", "_channels.tsv"
        )
        assert channels_path.exists()

        channels_df = pd.read_csv(channels_path, sep="\t")
        channel_names = list(channels_df["name"])
        assert channel_names
        assert "timestamp" not in channel_names
        assert "timeSinceStartup" not in channel_names
        assert "latency" in channel_names

        motion_df = pd.read_csv(motion_path, sep="\t", header=None, names=channel_names)
        assert list(motion_df.columns) == channel_names


def test_pipeline_real_csv_row_count_parity(real_csv_pipeline_output):
    """Each output motion.tsv keeps source row count (no drop/duplication)."""
    out = real_csv_pipeline_output
    source_rows = len(out.source_df)

    for motion_path in sorted(out.raw_motion_dir.glob("*_motion.tsv")):
        out_rows = len(pd.read_csv(motion_path, sep="\t", header=None))
        assert out_rows == source_rows

    for motion_path in sorted(out.deriv_motion_dir.glob("*_motion.tsv")):
        out_rows = len(pd.read_csv(motion_path, sep="\t", header=None))
        assert out_rows == source_rows


def test_pipeline_real_csv_system_completeness(real_csv_pipeline_output):
    """Real CSV produces expected core systems in output."""
    out = real_csv_pipeline_output
    systems = {
        _extract_system_from_motion_filename(path)
        for path in out.raw_motion_dir.glob("*_motion.tsv")
    }
    assert {"Head", "Hands", "Eyes"}.issubset(systems)


def test_pipeline_raw_derivative_schema_bijection_all_systems(real_csv_pipeline_output):
    """For every system, raw/derivative motion and channels schemas stay in sync."""
    out = real_csv_pipeline_output

    raw_motion_files = sorted(out.raw_motion_dir.glob("*_motion.tsv"))
    deriv_motion_files = sorted(out.deriv_motion_dir.glob("*_motion.tsv"))
    assert raw_motion_files
    assert len(raw_motion_files) == len(deriv_motion_files)

    for raw_motion in raw_motion_files:
        system = _extract_system_from_motion_filename(raw_motion)
        raw_channels = out.raw_motion_dir / raw_motion.name.replace("_motion.tsv", "_channels.tsv")
        deriv_motion = out.deriv_motion_dir / raw_motion.name
        deriv_channels = out.deriv_motion_dir / raw_motion.name.replace(
            "_motion.tsv", "_channels.tsv"
        )

        assert raw_channels.exists()
        assert deriv_motion.exists()
        assert deriv_channels.exists()

        raw_names = list(pd.read_csv(raw_channels, sep="\t")["name"])
        deriv_names = list(pd.read_csv(deriv_channels, sep="\t")["name"])

        assert raw_names == deriv_names, f"Schema drift between raw/derivative for {system}"

        raw_df = pd.read_csv(raw_motion, sep="\t", header=None, names=raw_names)
        deriv_df = pd.read_csv(deriv_motion, sep="\t", header=None, names=deriv_names)

        assert list(raw_df.columns) == raw_names
        assert list(deriv_df.columns) == deriv_names


def test_csv_source_columns_all_present_in_tsv_output(real_csv_pipeline_output):
    """Every data column from the source CSV appears in exactly one system's TSV output.

    The pipeline transforms timeSinceStartup → latency and routes every other
    column to a tracking-system TSV via prefix matching.  This test reads the
    original CSV header, collects all channel names from the output channels.tsv
    files, and asserts full coverage: no source data column is silently dropped.
    """
    out = real_csv_pipeline_output

    source_cols = set(out.source_df.columns)
    # timeSinceStartup is consumed by the pipeline (renamed to timestamp,
    # then converted to latency); it is not expected in the output.
    source_data_cols = source_cols - {"timeSinceStartup"}

    # Collect every channel name across all systems' channels.tsv
    all_output_channels: set[str] = set()
    channels_files = sorted(out.raw_motion_dir.glob("*_channels.tsv"))
    assert channels_files, "No channels.tsv files found in output"

    for ch_path in channels_files:
        names = list(pd.read_csv(ch_path, sep="\t")["name"])
        all_output_channels.update(names)

    # latency / latency_global are derived columns, not from the source CSV
    derived_cols = {"latency", "latency_global"}
    output_data_cols = all_output_channels - derived_cols

    missing = source_data_cols - output_data_cols
    assert not missing, f"Source CSV columns missing from TSV output: {sorted(missing)}"


def test_csv_data_values_preserved_in_tsv_output(real_csv_pipeline_output):
    """Data column values in the output TSV match the source CSV cell-for-cell.

    Non-time data columns pass through the pipeline with no transformation.
    Full float64 precision is preserved (no rounding), so this test asserts
    exact equality for numeric values and exact string equality for text.
    """
    out = real_csv_pipeline_output
    source_df = out.source_df

    derived_cols = {"latency", "latency_global"}
    channels_files = sorted(out.raw_motion_dir.glob("*_channels.tsv"))
    assert channels_files

    checked_columns = 0

    for ch_path in channels_files:
        channels_df = pd.read_csv(ch_path, sep="\t")
        channel_names = list(channels_df["name"])

        motion_path = ch_path.parent / ch_path.name.replace("_channels.tsv", "_motion.tsv")
        tsv_df = pd.read_csv(motion_path, sep="\t", header=None, names=channel_names)

        data_cols = [c for c in channel_names if c not in derived_cols]

        for col in data_cols:
            if col not in source_df.columns:
                continue

            src_vals = source_df[col].values
            tsv_vals = tsv_df[col].values

            assert len(src_vals) == len(tsv_vals), (
                f"Row count mismatch for {col}: source={len(src_vals)}, tsv={len(tsv_vals)}"
            )

            is_numeric = pd.api.types.is_numeric_dtype(source_df[col])

            if is_numeric:
                both_nan = pd.isna(src_vals) & pd.isna(tsv_vals)
                src_nan_only = pd.isna(src_vals) & ~pd.isna(tsv_vals)
                tsv_nan_only = ~pd.isna(src_vals) & pd.isna(tsv_vals)

                assert not src_nan_only.any(), f"{col}: source has NaN where TSV does not"
                assert not tsv_nan_only.any(), f"{col}: TSV has NaN where source does not"

                valid = ~both_nan
                if valid.any():
                    np.testing.assert_array_equal(
                        tsv_vals[valid].astype(float),
                        src_vals[valid].astype(float),
                        err_msg=f"Value mismatch in column {col}",
                    )
            else:
                src_series = source_df[col]
                tsv_series = tsv_df[col]

                src_na = src_series.isna()
                tsv_na = tsv_series.isna()
                assert (src_na == tsv_na).all(), f"NaN position mismatch in string column {col}"

                valid = ~src_na
                if valid.any():
                    src_str = src_series[valid].astype(str).values
                    tsv_str = tsv_series[valid].astype(str).values
                    mismatched = src_str != tsv_str
                    assert not mismatched.any(), (
                        f"String mismatch in column {col} at rows "
                        f"{np.where(mismatched)[0][:5].tolist()}"
                    )

            checked_columns += 1

    assert checked_columns > 0, "No data columns were checked"


def test_sourcedata_is_verbatim_copy(tmp_path, minimal_config_dict):
    """After a run, sourcedata/ holds a verbatim copy of the source CSV."""
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    src = _write_session_dir(data_dir, "sess_sd", session_id="SD", with_events=True)
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[{"source_dir": "sess_sd", "subject_id": "01", "session_label": "01"}],
    )
    run(str(cfg))
    copied = bids_root / "sourcedata" / "sub-01" / "ses-01" / "SD_ContinuousData.csv"
    assert copied.exists()
    assert copied.read_bytes() == (src / "SD_ContinuousData.csv").read_bytes()


def test_events_at_session_root_with_custom_class(tmp_path, minimal_config_dict):
    """A custom class CSV is merged into the session-root events file."""
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    src = _write_session_dir(data_dir, "sess_cc", session_id="CC", with_events=True)
    custom = src / "CC_CustomTables"
    custom.mkdir()
    pd.DataFrame({"onset": [0.25], "duration": [0.0], "reaction_time": [0.3]}).to_csv(
        custom / "CC_ChoiceEvent.csv", index=False
    )
    (custom / "CC_CustomTables.json").write_text(
        '{"CustomTables":{"ChoiceEvent":{"RowCount":1,"Columns":'
        '{"reaction_time":{"Description":"RT","Units":"s","Format":"float"}}}}}'
    )
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[{"source_dir": "sess_cc", "subject_id": "01", "session_label": "01"}],
    )
    run(str(cfg))
    events = pd.read_csv(
        bids_root / "sub-01" / "ses-01" / "sub-01_ses-01_task-vr_events.tsv", sep="\t"
    )
    assert "reaction_time" in events.columns
    assert "ChoiceEvent" in list(events["name"])


def test_events_and_motion_share_the_main_data_time_zero(tmp_path, minimal_config_dict, caplog):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    bad_dir = _write_session_dir(data_dir, "sess_b", session_id="B")
    bad = pd.read_csv(bad_dir / "B_ContinuousData.csv")
    hardware = bad["timeSinceStartup"] + 100.0
    bad = bad.assign(Node_Head_Time=hardware, Node_HandLeft_Time=hardware, timeSinceStartup=0.0)
    bad.to_csv(bad_dir / "B_ContinuousData.csv", index=False)
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    session_dir = _write_session_dir(data_dir, "sess_a", session_id="A")
    metadata_path = session_dir / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["face_enabled"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    face = pd.DataFrame({"timeSinceStartup": [1.011111, 1.022222], "Eyes_Closed_L": [0.0, 0.0]})
    face.to_csv(session_dir / "A_FaceExpressionData.csv", index=False)
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_b", "subject_id": "02", "session_label": "01"},
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
        ],
    )
    run(str(cfg))
    assert "no valid timeSinceStartup sample" in caplog.text
    assert not (bids_root / "sub-02").exists()
    session_out = bids_root / "sub-01" / "ses-01"
    events = pd.read_csv(session_out / "sub-01_ses-01_task-vr_events.tsv", sep="\t")
    assert list(events["onset"]) == pytest.approx([-1.0, -0.5, 0.0])
    for system, first_latency in (("Head", 0.0), ("Face", 0.011111)):
        (motion_path,) = (session_out / "motion").glob(f"*_tracksys-{system}_motion.tsv")
        motion = pd.read_csv(motion_path, sep="\t", header=None)
        assert motion[0].iloc[0] == pytest.approx(first_latency)


def test_masking_blanks_hand_and_gaze_measurements_only(tmp_path, minimal_config_dict):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    session_dir = _write_session_dir(data_dir, "sess_m", session_id="M", with_events=False)
    t = 1.0 + np.arange(8) / 10.0
    flagged = np.array([False, False, True, True, False, False, False, False])
    values = np.arange(8) + 0.5
    continuous = {
        "timeSinceStartup": t,
        "Node_HandLeft_Time": t + 100.0,
        "LeftHand_Root_px": values,
        "LeftHand_Status_HandTracked": (~flagged).astype(int),
        "Eyes_Time": t + 200.0,
        "LeftEye_GazeOrigin_x": values,
        "Node_EyeCenter_px": values,
    }
    pd.DataFrame(continuous).to_csv(session_dir / "M_ContinuousData.csv", index=False)
    closed = np.where(flagged, 0.95, 0.0)
    face = {"timeSinceStartup": t, "Face_Time": t + 300.0, "Eyes_Closed_L": closed}
    face["Eyes_Closed_R"] = closed
    pd.DataFrame(face).to_csv(session_dir / "M_FaceExpressionData.csv", index=False)
    metadata_path = session_dir / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(face_enabled=True, eyes_enabled=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    minimal_config_dict["validation"] = {
        "enabled_checks": ["hands_tracking_loss", "eyes_closed"],
        "settings": {"eyes_closed_use_min_duration": False},
    }
    minimal_config_dict["preprocessing"] = {
        "apply_quality_masking": True,
        "alternate_time_columns": {
            "Hands": "Node_HandLeft_Time",
            "Eyes": "Eyes_Time",
            "Face": "Face_Time",
        },
    }
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[{"source_dir": "sess_m", "subject_id": "01", "session_label": "01"}],
    )
    run(str(cfg))
    motion_dir = bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01" / "motion"
    out = {}
    for system in ("Hands", "Eyes", "Face"):
        (motion_path,) = motion_dir.glob(f"*_tracksys-{system}_motion.tsv")
        channels = pd.read_csv(str(motion_path).replace("_motion.tsv", "_channels.tsv"), sep="\t")
        out[system] = pd.read_csv(motion_path, sep="\t", header=None, names=list(channels["name"]))
    assert list(out["Hands"]["LeftHand_Root_px"].isna()) == list(flagged)
    assert list(out["Eyes"]["LeftEye_GazeOrigin_x"].isna()) == list(flagged)
    assert not out["Hands"]["LeftHand_Status_HandTracked"].isna().any()
    assert not out["Eyes"]["Node_EyeCenter_px"].isna().any()
    assert not out["Face"].isna().any().any()


def test_pipeline_scans_acq_time_keeps_subseconds_and_utc(tmp_path, minimal_config_dict):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    session_dir = _write_session_dir(data_dir, "sess_a", session_id="A")
    metadata_path = session_dir / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["utc_start_iso8601"] = "2026-06-10T10:14:19.4723370Z"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    cfg_path = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
        ],
    )

    run(str(cfg_path))

    for session_dir in (
        bids_root / "sub-01" / "ses-01",
        bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01",
    ):
        (scans_path,) = session_dir.glob("*_scans.tsv")
        scans = pd.read_csv(scans_path, sep="\t", dtype=str)
        assert set(scans["acq_time"]) == {"2026-06-10T10:14:19.472337Z"}


def test_hand_joint_channels_carry_the_detected_frame(tmp_path, minimal_config_dict, caplog):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    session_dir = _write_session_dir(data_dir, "sess_h", session_id="H")
    continuous = pd.read_csv(session_dir / "H_ContinuousData.csv")
    root = {"py": 1.0, "pz": 0.2, "qx": 0.1, "qy": 0.2, "qz": 0.3, "qw": 0.927}
    wrist = {"x": continuous["LeftHand_Root_px"], "y": 1.0, "z": -0.2, "qx": -0.1, "qy": -0.2}
    wrist.update(qz=0.3, qw=0.927)
    continuous = continuous.assign(
        **{f"LeftHand_Root_{key}": value for key, value in root.items()},
        **{f"Left_XRHand_Wrist_{key}": value for key, value in wrist.items()},
    )
    continuous.to_csv(session_dir / "H_ContinuousData.csv", index=False)
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[{"source_dir": "sess_h", "subject_id": "01", "session_label": "01"}],
    )
    with caplog.at_level(logging.INFO, logger="resxr"):
        run(str(cfg))
    assert "Hand joint frame in H: Left tracking_space" in caplog.text
    for session_out in (bids_root, bids_root / "derivatives" / "resxr"):
        motion_dir = session_out / "sub-01" / "ses-01" / "motion"
        (channels_path,) = motion_dir.glob("*_tracksys-Hands_channels.tsv")
        channels = pd.read_csv(channels_path, sep="\t")
        frames = dict(zip(channels["name"], channels["reference_frame"], strict=True))
        assert frames["Left_XRHand_Wrist_z"] == "tracking_space"
        assert frames["LeftHand_Root_pz"] == "global"
        sidecar = json.loads(channels_path.with_suffix(".json").read_text(encoding="utf-8"))
        assert set(sidecar["reference_frame"]["Levels"]) == {"global", "tracking_space"}
        (head_json,) = motion_dir.glob("*_tracksys-Head_channels.json")
        head_sidecar = json.loads(head_json.read_text(encoding="utf-8"))
        assert set(head_sidecar["reference_frame"]["Levels"]) == {"global"}
