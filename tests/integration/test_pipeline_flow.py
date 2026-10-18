"""Integration tests for full pipeline orchestration."""

from __future__ import annotations

import json
import logging
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import resxr.pipeline as pipeline_module
from resxr.bids.layout import BIDSLayout
from resxr.cli import main
from resxr.core.config import PipelineConfig, SessionMapping
from resxr.core.exceptions import DataLoadError
from resxr.pipeline import _SessionOutputs, process_session_from_mapping, run

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
    report: bool = False,
) -> Path:
    cfg = dict(minimal_config_dict)
    cfg["input"] = dict(cfg["input"])
    cfg["output"] = dict(cfg["output"])
    cfg["report"] = dict(cfg["report"])

    cfg["input"]["data_dir"] = str(data_dir)
    cfg["output"]["bids_root"] = str(bids_root)
    cfg["report"]["enabled"] = report
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


def _edit_metadata(session_dir: Path, edit) -> Path:
    """Apply *edit* (which changes the dict in place) to the session's metadata file."""
    path = session_dir / "session_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    edit(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _zero_hands_clock(session_dir: Path) -> None:
    _edit_continuous(session_dir, lambda df: df.assign(Node_HandLeft_Time=0.0))


def _zero_main_clock_without_head_clock(session_dir: Path) -> None:
    _edit_continuous(
        session_dir, lambda df: df.drop(columns="Node_Head_Time").assign(timeSinceStartup=0.0)
    )


def _put_text(column: str):
    def edit(df: pd.DataFrame) -> pd.DataFrame:
        values = df[column].astype(object)
        values.iloc[1] = "abc"
        return df.assign(**{column: values})

    return edit


def _text_in_main_clock(session_dir: Path) -> None:
    _edit_continuous(session_dir, _put_text("timeSinceStartup"))


def _text_in_hands_clock(session_dir: Path) -> None:
    _edit_continuous(session_dir, _put_text("Node_HandLeft_Time"))


def _keep_one_row(session_dir: Path) -> None:
    _edit_continuous(session_dir, lambda df: df.iloc[:1])


def _add_face_file_without_clocks(session_dir: Path) -> None:
    _edit_metadata(session_dir, lambda m: m.update(face_enabled=True))
    face = pd.DataFrame({"Eyes_Closed_L": [0.0, 0.0, 0.0]})
    face.to_csv(session_dir / "BAD_FaceExpressionData.csv", index=False)


@pytest.mark.parametrize(
    ("make_untimeable", "message"),
    [
        (_zero_hands_clock, "Stream Hands: no non-zero timestamp found"),
        (_zero_main_clock_without_head_clock, "Stream Head: no non-zero timestamp found"),
        (_text_in_main_clock, "Stream Head: non-numeric value in 'timeSinceStartup' column"),
        (_text_in_hands_clock, "Stream Hands: non-numeric value in 'timestamp' column"),
        (_keep_one_row, "Stream Head: need at least 2 rows"),
        (_add_face_file_without_clocks, "No timestamp column found in "),
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
            session_dir,
            lambda df: df.assign(
                Node_Head_Time=df["timeSinceStartup"] + 50.0,
                Node_HandLeft_Time=df["timeSinceStartup"] + 100.0,
            ),
        )
    make_untimeable(data_dir / "sess_bad")
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
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


def test_the_derivative_dataset_has_the_configured_license(tmp_path, minimal_config_dict):
    data_dir = tmp_path / "sessions"
    bids_root = tmp_path / "bids_out"
    _write_session_dir(data_dir, "sess_a", session_id="A")
    minimal_config_dict["bids"]["license"] = "CC-BY-4.0"
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

    raw = json.loads((bids_root / "dataset_description.json").read_text(encoding="utf-8"))
    deriv_path = bids_root / "derivatives" / "resxr" / "dataset_description.json"
    deriv = json.loads(deriv_path.read_text(encoding="utf-8"))
    assert raw["License"] == deriv["License"] == "CC-BY-4.0"


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
    _edit_metadata(session_dir, lambda m: m.update(face_enabled=True))
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
    _edit_metadata(session_dir, lambda m: m.update(face_enabled=True, eyes_enabled=True))
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
    _edit_metadata(
        session_dir, lambda m: m.update(utc_start_iso8601="2026-06-10T10:14:19.4723370Z")
    )

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


# ── Run outcome: end-of-run summary and exit code ───────────────────


def _write_clocked_session_dir(data_dir: Path, name: str, *, session_id: str) -> Path:
    """A valid session with per-system clocks, so every timing check can run."""
    session_dir = _write_session_dir(data_dir, name, session_id=session_id)
    _edit_continuous(
        session_dir,
        lambda df: df.assign(
            Node_Head_Time=df["timeSinceStartup"] + 50.0,
            Node_HandLeft_Time=df["timeSinceStartup"] + 100.0,
        ),
    )
    return session_dir


def _two_session_config(tmp_path, minimal_config_dict, *, missing_second=False):
    """Two clocked sessions mapped to sub-01 and sub-02 (the second folder absent if asked)."""
    data_dir = tmp_path / "sessions"
    _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    second = "sess_missing" if missing_second else "sess_b"
    if not missing_second:
        _write_clocked_session_dir(data_dir, second, session_id="B")
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    return data_dir, _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
            {"source_dir": second, "subject_id": "02", "session_label": "01"},
        ],
    )


def test_run_returns_every_session_and_exit_code_0_when_all_processed(
    tmp_path, minimal_config_dict, caplog
):
    _, cfg = _two_session_config(tmp_path, minimal_config_dict)
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [(s.subject_id, s.processed) for s in result.sessions] == [("01", True), ("02", True)]
    assert result.exit_code == 0
    summary = [r for r in caplog.records if r.getMessage().startswith("Run summary:")]
    assert [r.levelname for r in summary] == ["INFO"]
    assert summary[0].getMessage() == "Run summary: 2 session(s) found, 2 processed, 0 skipped."
    assert "  processed sess_a (sub-01 ses-01): 0 warning(s), 0 error(s)" in caplog.text
    assert f"Pipeline complete. Output at: {tmp_path / 'bids_out'}" in caplog.text
    assert main(["-c", str(cfg)]) == 0


def test_a_skipped_session_gives_a_warning_summary_and_exit_code_1(
    tmp_path, minimal_config_dict, caplog
):
    _, cfg = _two_session_config(tmp_path, minimal_config_dict, missing_second=True)
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [True, False]
    assert "Session directory not found" in result.skipped[0].reason
    assert result.exit_code == 1
    summary = [r for r in caplog.records if r.getMessage().startswith("Run summary:")]
    assert [r.levelname for r in summary] == ["WARNING"]
    assert "Run summary: 2 session(s) found, 1 processed, 1 skipped." in caplog.text
    assert "  skipped sess_missing (sub-02 ses-01): Session directory not found" in caplog.text
    assert main(["-c", str(cfg)]) == 1


def test_quiet_run_still_shows_a_skipped_session_and_exits_1(tmp_path, minimal_config_dict, capsys):
    _, cfg = _two_session_config(tmp_path, minimal_config_dict, missing_second=True)
    resxr_logger = logging.getLogger("resxr")
    level, handlers = resxr_logger.level, list(resxr_logger.handlers)
    try:
        assert main(["-q", "-c", str(cfg)]) == 1
    finally:  # main() set the 'resxr' logger to ERROR with its own handler
        resxr_logger.setLevel(level)
        resxr_logger.handlers[:] = handlers
    out = capsys.readouterr().out
    assert (
        "ERROR - resxr.pipeline - Failed to process sess_missing: Session directory not found"
    ) in out
    assert " INFO - " not in out
    assert " WARNING - " not in out


def test_every_session_failing_gives_an_error_summary_and_exit_code_1(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    for name in ("sess_a", "sess_b"):
        (data_dir / name / "session_metadata.json").unlink()
    assert main(["-c", str(cfg)]) == 1
    summary = [r for r in caplog.records if r.getMessage().startswith("Run summary:")]
    assert [r.levelname for r in summary] == ["ERROR"]
    assert "Run summary: 2 session(s) found, 0 processed, 2 skipped." in caplog.text
    assert not (tmp_path / "bids_out" / "dataset_description.json").exists()
    assert "Pipeline complete" not in caplog.text  # no output was written


def _fail_session_a(monkeypatch, error: BaseException) -> None:
    """Make the split of session 'A' raise *error*; other sessions are split as usual."""
    split = pipeline_module.split_continuous_data

    def split_or_fail(session, *args, **kwargs):
        if session.session_id == "A":
            raise error
        return split(session, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "split_continuous_data", split_or_fail)


def test_an_unexpected_error_skips_only_that_session(
    tmp_path, minimal_config_dict, caplog, monkeypatch
):
    _, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _fail_session_a(monkeypatch, ValueError("boom"))
    with caplog.at_level(logging.DEBUG, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert result.skipped[0].reason == "unexpected error: ValueError: boom"
    assert result.exit_code == 1
    (failed,) = [r for r in caplog.records if r.getMessage().startswith("Failed to process")]
    assert failed.levelname == "ERROR"
    assert failed.getMessage() == "Failed to process sess_a: unexpected error: ValueError: boom"
    (traceback,) = [r for r in caplog.records if r.exc_info]
    assert traceback.levelname == "DEBUG"
    assert traceback.exc_info[0] is ValueError
    assert "  skipped sess_a (sub-01 ses-01): unexpected error: ValueError: boom" in caplog.text
    assert list((tmp_path / "bids_out" / "sub-02" / "ses-01" / "motion").glob("*_motion.tsv"))
    assert main(["-c", str(cfg)]) == 1


def test_keyboard_interrupt_in_a_session_still_stops_the_run(
    tmp_path, minimal_config_dict, monkeypatch
):
    _, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _fail_session_a(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(str(cfg))
    assert not (tmp_path / "bids_out" / "sub-02").exists()
    assert main(["-c", str(cfg)]) == 130


def test_checks_not_run_appear_in_the_summary(tmp_path, minimal_config_dict, caplog):
    data_dir = tmp_path / "sessions"
    session_dir = _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    _edit_continuous(session_dir, lambda df: df.drop(columns="RightHand_Status_HandTracked"))
    minimal_config_dict["validation"]["enabled_checks"] = ["hands_tracking_loss"]
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[{"source_dir": "sess_a", "subject_id": "01", "session_label": "01"}],
    )
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert result.exit_code == 0  # the session was processed; the gap is reported
    (outcome,) = result.sessions
    assert outcome.warnings >= 1
    assert [c.check for c in outcome.checks_not_run] == ["hands_tracking_loss"]
    summary = [r for r in caplog.records if r.getMessage().startswith("Run summary:")]
    assert [r.levelname for r in summary] == ["WARNING"]
    assert (
        "checks not run: hands_tracking_loss on Hands: right_hand: column "
        "RightHand_Status_HandTracked missing"
    ) in caplog.text


def _run_with_every_stream_check(
    tmp_path, minimal_config_dict, caplog, *, edit_metadata, face_file=False, systems=None
):
    """Run one clocked session with the stream checks enabled and the report on.

    Returns the session's outcome, the run-summary records and the report text.
    """
    data_dir = tmp_path / "sessions"
    session_dir = _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    _edit_metadata(session_dir, edit_metadata)
    if face_file:
        pd.DataFrame(
            {
                "timeSinceStartup": [1.0, 1.033333, 1.066667],
                "Eyes_Closed_L": [0.0, 0.0, 0.0],
                "Eyes_Closed_R": [0.0, 0.0, 0.0],
            }
        ).to_csv(session_dir / "A_FaceExpressionData.csv", index=False)
    minimal_config_dict["validation"]["enabled_checks"] = [
        "sampling_rate",
        "stats_summary",
        "hands_tracking_loss",
        "eyes_closed",
    ]
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    if systems:
        minimal_config_dict["systems"] = systems
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[{"source_dir": "sess_a", "subject_id": "01", "session_label": "01"}],
        report=True,
    )
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    (outcome,) = result.sessions
    summary = [r for r in caplog.records if r.getMessage().startswith("Run summary:")]
    report = tmp_path / "bids_out" / "sub-01" / "ses-01" / "A_report.html"
    return outcome, summary, report.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("edit_metadata", "face_file", "systems", "info_line"),
    [
        pytest.param(
            lambda m: m.update(face_enabled=True),
            True,
            {"Face": {"enabled": False}},
            "Session 'A': Face not exported: disabled in the config (systems.Face.enabled); "
            "the face file stays only in sourcedata/",
            id="face_disabled_in_config",
        ),
        pytest.param(
            lambda m: m.update(face_enabled=False), False, None, None, id="face_enabled_false"
        ),
        pytest.param(
            lambda m: m.update(hands_enabled=False),
            False,
            None,
            "Session 'A': Hands not exported: hands_enabled is false in the metadata",
            id="hands_enabled_false",
        ),
    ],
)
def test_a_stream_turned_off_on_purpose_is_not_a_check_not_run(
    tmp_path, minimal_config_dict, caplog, edit_metadata, face_file, systems, info_line
):
    outcome, summary, report = _run_with_every_stream_check(
        tmp_path,
        minimal_config_dict,
        caplog,
        edit_metadata=edit_metadata,
        face_file=face_file,
        systems=systems,
    )
    assert outcome.processed
    assert outcome.checks_not_run == []
    assert [r.levelname for r in summary] == ["INFO"]
    assert "not run" not in caplog.text
    assert "Checks not run" not in report
    if info_line:
        assert info_line in caplog.text  # the stream's own INFO line stays


@pytest.mark.parametrize(
    ("edit_metadata", "why"),
    [
        pytest.param(
            lambda m: m.update(face_enabled=True),
            "metadata: face_enabled is true; no face file was loaded",
            id="face_enabled_true_no_face_file",
        ),
        pytest.param(
            lambda m: m.pop("face_enabled"),
            "metadata: face_enabled missing; no face file",
            id="face_enabled_missing",
        ),
        pytest.param(
            lambda m: m.update(face_enabled=0),
            "metadata: face_enabled is 0, not true or false; no face file",
            id="face_enabled_0",
        ),
        pytest.param(
            lambda m: m.update(face_enabled="false"),
            "metadata: face_enabled is text, read as false; no face file",
            id="face_enabled_string_false",
        ),
        pytest.param(
            lambda m: m.update(face_enabled=None),
            "metadata: face_enabled is null, not true or false; no face file",
            id="face_enabled_null",
        ),
    ],
)
def test_an_expected_stream_that_is_absent_is_a_check_not_run(
    tmp_path, minimal_config_dict, caplog, edit_metadata, why
):
    outcome, summary, report = _run_with_every_stream_check(
        tmp_path, minimal_config_dict, caplog, edit_metadata=edit_metadata
    )
    reason = f"the session has no Face stream ({why})"
    assert [(c.check, c.stream, c.reason) for c in outcome.checks_not_run] == [
        ("eyes_closed", "Face", reason)
    ]
    assert [r.levelname for r in summary] == ["WARNING"]
    assert f"Session 'A': check 'eyes_closed' not run: {reason}" in caplog.text
    assert "<h2>Checks not run (1)</h2>" in report


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        pytest.param(
            lambda df: df.drop(
                columns=[c for c in df.columns if c.startswith(("Left", "Right", "Node_Hand"))]
            ),
            "metadata: hands_enabled is true; no Hands columns in the continuous data",
            id="enabled_without_columns",
        ),
        pytest.param(
            lambda df: df.drop(columns="Node_HandLeft_Time"),
            "metadata: hands_enabled missing; its 4 columns were not exported",
            id="key_missing_with_columns",
        ),
    ],
)
def test_the_not_run_reason_says_why_the_stream_is_absent(
    tmp_path, minimal_config_dict, caplog, edit, why
):
    data_dir = tmp_path / "sessions"
    session_dir = _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    _edit_continuous(session_dir, edit)
    if why.startswith("metadata: hands_enabled missing"):
        _edit_metadata(session_dir, lambda m: m.pop("hands_enabled"))
    minimal_config_dict["validation"]["enabled_checks"] = ["hands_tracking_loss"]
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {"Head": "Node_Head_Time"}
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[{"source_dir": "sess_a", "subject_id": "01", "session_label": "01"}],
    )
    (outcome,) = run(str(cfg)).sessions
    assert [c.reason for c in outcome.checks_not_run] == [
        f"the session has no Hands stream ({why})"
    ]


@pytest.mark.parametrize("session_id", [None, ""], ids=["null", "empty"])
def test_a_null_or_empty_session_id_is_read_as_unknown(
    tmp_path, minimal_config_dict, caplog, session_id
):
    data_dir = tmp_path / "sessions"
    session_dir = _write_session_dir(data_dir, "sess_a", session_id="A")
    metadata_path = _edit_metadata(session_dir, lambda m: m.update(session_id=session_id))
    (session_dir / "A_CustomTables").mkdir()
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[{"source_dir": "sess_a", "subject_id": "01", "session_label": "01"}],
        report=True,
    )
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert result.exit_code == 0
    assert f"{metadata_path} has no session_id; using 'unknown': custom tables are read" in (
        caplog.text
    )
    assert (
        f"Custom tables folder(s) ['A_CustomTables'] in {session_dir} do not match the expected "
        "name 'unknown_CustomTables' (from session_id 'unknown')"
    ) in caplog.text
    assert (tmp_path / "bids_out" / "sub-01" / "ses-01" / "unknown_report.html").exists()


def test_nested_sessions_without_mappings_are_an_error_with_exit_code_1(
    tmp_path, minimal_config_dict, caplog
):
    data_dir = tmp_path / "DATA"
    _write_session_dir(data_dir / "Project" / "Task", "sess_a", session_id="A")
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[],
    )
    assert main(["-c", str(cfg)]) == 1
    assert f"No sessions found under {data_dir}: without session_mappings" in caplog.text
    assert "1 deeper folder(s) contain one" in caplog.text
    assert str(data_dir / "Project" / "Task" / "sess_a") in caplog.text
    assert "Run summary: 0 session(s) found, 0 processed, 0 skipped." in caplog.text
    assert "Pipeline complete" not in caplog.text


def test_header_only_continuous_file_skips_the_session_with_an_error(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_continuous(data_dir / "sess_a", lambda df: df.iloc[:0])
    result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert (
        "Failed to process sess_a: No tracking stream to export from "
        f"{data_dir / 'sess_a'}: the continuous data file has no data rows"
    ) in caplog.text
    assert result.exit_code == 1


def test_continuous_file_without_tracking_columns_skips_the_session_with_an_error(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_continuous(data_dir / "sess_a", lambda df: df[["timeSinceStartup"]].assign(EDA_uS=0.5))
    result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert (
        "Failed to process sess_a: No tracking stream to export from "
        f"{data_dir / 'sess_a'}: no enabled tracking system has data columns (see the "
        "messages above)"
    ) in caplog.text
    assert result.exit_code == 1


def test_events_columns_without_a_description_are_listed(tmp_path, minimal_config_dict, caplog):
    data_dir = tmp_path / "sessions"
    session_dir = _write_session_dir(data_dir, "sess_e", session_id="E")
    events_path = session_dir / "E_EventsData.csv"
    pd.read_csv(events_path).assign(stim_file="img.png", block=1).to_csv(events_path, index=False)
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=[{"source_dir": "sess_e", "subject_id": "01", "session_label": "01"}],
    )
    run(str(cfg))
    assert (
        "Session 'E': events.tsv columns not described in events.json: stim_file, block"
    ) in caplog.text
    events = pd.read_csv(
        tmp_path / "bids_out" / "sub-01" / "ses-01" / "sub-01_ses-01_task-vr_events.tsv", sep="\t"
    )
    assert {"stim_file", "block"} <= set(events.columns)  # still passed through


# ── A skipped session leaves nothing in the output (D9) ──────────────


def _session_outputs(bids_root: Path, sub: str) -> list[Path]:
    """Every sourcedata, raw and derivative path of one subject that exists."""
    trees = (bids_root / "sourcedata" / sub, bids_root / sub)
    trees += (bids_root / "derivatives" / "resxr" / sub,)
    return [tree for tree in trees if tree.exists()]


def _files(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def test_a_session_skipped_at_the_split_leaves_no_output(tmp_path, minimal_config_dict, caplog):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _zero_hands_clock(data_dir / "sess_a")  # as 4 Oct case S3: skipped after its sourcedata copy
    bids_root = tmp_path / "bids_out"
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert _session_outputs(bids_root, "sub-01") == []
    (removed,) = [r for r in caplog.records if r.getMessage().startswith("Removed the output")]
    assert removed.levelname == "INFO"
    assert removed.getMessage() == (  # sourcedata/ was created by sess_a alone (M10)
        "Removed the output written for skipped session sess_a: sourcedata"
    )
    assert _files(bids_root / "sourcedata" / "sub-02" / "ses-01") == _files(data_dir / "sess_b")
    assert list((bids_root / "sub-02" / "ses-01" / "motion").glob("*_motion.tsv"))
    deriv = bids_root / "derivatives" / "resxr" / "sub-02" / "ses-01" / "motion"
    assert list(deriv.glob("*_motion.tsv"))
    assert _files(data_dir / "sess_a")  # the input is untouched


def _fail_report_of_session_a(monkeypatch) -> None:
    """Write session A's report, then raise, as a late error in the report step would."""
    generate = pipeline_module.ReportGenerator.generate

    def generate_or_fail(self, session, output_path, *args, **kwargs):
        written = generate(self, session, output_path, *args, **kwargs)
        if session.session_id == "A":
            raise TypeError("must be real number, not str")
        return written

    monkeypatch.setattr(pipeline_module.ReportGenerator, "generate", generate_or_fail)


@pytest.mark.parametrize("report_dir", [None, "reports"], ids=["report_in_session", "report_dir"])
def test_a_late_failure_removes_every_file_of_that_session(
    tmp_path, minimal_config_dict, caplog, monkeypatch, report_dir
):
    data_dir = tmp_path / "sessions"
    _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    _write_clocked_session_dir(data_dir, "sess_b", session_id="B")
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    if report_dir is not None:
        minimal_config_dict["report"]["output_dir"] = str(tmp_path / report_dir)
    bids_root = tmp_path / "bids_out"
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
            {"source_dir": "sess_b", "subject_id": "02", "session_label": "01"},
        ],
        report=True,
    )
    _fail_report_of_session_a(monkeypatch)
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert result.skipped[0].reason == "unexpected error: TypeError: must be real number, not str"
    assert _session_outputs(bids_root, "sub-01") == []
    reports = tmp_path / report_dir if report_dir else bids_root / "sub-02" / "ses-01"
    assert [p.name for p in reports.rglob("*_report.html")] == ["B_report.html"]
    shown = ["derivatives", "sourcedata"]  # created by session A alone, so removed (M10)
    shown += [str(tmp_path / report_dir / "A_report.html")] if report_dir else []
    shown.append("sub-01")
    assert (
        f"Removed the output written for skipped session sess_a: {', '.join(sorted(shown))}"
    ) in caplog.text
    assert list((bids_root / "sub-02" / "ses-01" / "motion").glob("*_motion.tsv"))
    assert (bids_root / "sourcedata" / "sub-02" / "ses-01").exists()
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    assert list(participants["participant_id"]) == ["sub-02"]


def test_a_session_folder_that_existed_before_is_kept_with_a_warning(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    bids_root = tmp_path / "bids_out"
    run(str(cfg))  # a complete first run (output.overwrite is true in this config)
    before = {tree: _files(tree) for tree in _session_outputs(bids_root, "sub-01")}
    assert len(before) == 3
    _zero_hands_clock(data_dir / "sess_a")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    after = {tree: _files(tree) for tree in _session_outputs(bids_root, "sub-01")}
    assert after.keys() == before.keys()  # every folder kept
    sourcedata_tree = bids_root / "sourcedata" / "sub-01"
    for tree in before.keys() - {sourcedata_tree}:  # raw and derivative files kept as they were
        assert after[tree] == before[tree]
    # the sourcedata copy was redone (output.overwrite) before the session failed
    assert after[sourcedata_tree].keys() == before[sourcedata_tree].keys()
    assert after[sourcedata_tree] != before[sourcedata_tree]
    sourcedata = bids_root / "sourcedata" / "sub-01" / "ses-01"
    kept = [r for r in caplog.records if " rerun." in r.getMessage()]
    assert [r.levelname for r in kept] == ["WARNING"] * 3
    assert [r.getMessage() for r in kept] == [
        f"{sourcedata} existed before this session started and may now hold partial files "
        "from the skipped session sess_a; remove it or fix the session and rerun.",
    ] + [  # not written in this run, but stale (M16)
        f"{folder} holds output of an earlier run for a session that was skipped now "
        "(sess_a); remove it or fix the session and rerun."
        for folder in (
            bids_root / "sub-01" / "ses-01",
            bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01",
        )
    ]
    assert "Removed the output" not in caplog.text


def test_a_session_skipped_while_loading_writes_and_removes_nothing(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    (data_dir / "sess_a" / "session_metadata.json").unlink()
    bids_root = tmp_path / "bids_out"
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert _session_outputs(bids_root, "sub-01") == []
    assert "Removed the output" not in caplog.text
    assert " rerun." not in caplog.text


# The report stays off here: on code without the rule, a report named after "M:evil" would be
# written to drive M:, outside tmp_path.
@pytest.mark.parametrize("session_id", ["a:b", "E:evil", "M:evil", "a?b", "a/b", "..\\x"])
def test_a_session_id_that_is_not_a_file_name_stem_skips_the_session_before_any_write(
    tmp_path, minimal_config_dict, caplog, session_id
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    metadata = _edit_metadata(data_dir / "sess_a", lambda m: m.update(session_id=session_id))
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert result.skipped[0].reason.startswith(
        f"{metadata}: session_id {json.dumps(session_id)} may hold only letters (A-Z, a-z), "
    )
    assert _session_outputs(tmp_path / "bids_out", "sub-01") == []
    assert "Removed the output" not in caplog.text  # nothing was written


def test_removal_never_reaches_the_input_or_outside_the_output(tmp_path, minimal_config_dict):
    minimal_config_dict["input"]["data_dir"] = str(tmp_path / "data")
    minimal_config_dict["output"]["bids_root"] = str(tmp_path / "data" / "out")
    config = PipelineConfig.model_validate(minimal_config_dict)
    bids = SimpleNamespace(bids_root=tmp_path / "data" / "out")
    source = tmp_path / "data" / "out" / "sub-09" / "ses-01"  # an input folder in the output
    outputs = _SessionOutputs(config, bids, [source])
    assert outputs._deletable(tmp_path / "data" / "out" / "sub-01")
    assert not outputs._deletable(tmp_path / "data" / "out")  # the BIDS root itself
    assert not outputs._deletable(tmp_path / "elsewhere" / "sub-01")
    assert not outputs._deletable(source)
    assert not outputs._deletable(source / "x.csv")
    assert not outputs._deletable(tmp_path / "data" / "out" / "sub-09")  # holds the input
    assert not outputs._deletable(tmp_path / "data")


def test_a_late_failure_in_an_existing_output_warns_once_per_folder(
    tmp_path, minimal_config_dict, caplog, monkeypatch
):
    data_dir = tmp_path / "sessions"
    _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    _write_clocked_session_dir(data_dir, "sess_b", session_id="B")
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    bids_root = tmp_path / "bids_out"
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
            {"source_dir": "sess_b", "subject_id": "02", "session_label": "01"},
        ],
        report=True,
    )
    run(str(cfg))  # a complete first run
    _fail_report_of_session_a(monkeypatch)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="resxr"):
        run(str(cfg))
    warned = [
        r.getMessage().split(" existed before")[0]
        for r in caplog.records
        if "existed before this session started and may now hold partial files from the "
        "skipped session sess_a; remove it or fix the session and rerun." in r.getMessage()
    ]
    assert warned == [  # the report inside sub-01/ses-01 gets no line of its own
        str(bids_root / "sourcedata" / "sub-01" / "ses-01"),
        str(bids_root / "sub-01" / "ses-01"),
        str(bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01"),
    ]
    assert (bids_root / "sub-01" / "ses-01" / "A_report.html").exists()
    assert "Removed the output" not in caplog.text


# ── Clock values between valid values (review 16 Oct) ─────────────────


def _write_long_clocked_session(data_dir: Path, name: str, *, session_id: str, n: int = 30):
    """A valid session of *n* regular rows with Head and Hands clocks."""
    session_dir = _write_session_dir(data_dir, name, session_id=session_id)
    t = 1.0 + np.arange(n) / 72.0
    _edit_continuous(
        session_dir,
        lambda df: pd.DataFrame(
            {
                "timeSinceStartup": t,
                "Node_Head_Time": t + 50.0,
                "Node_Head_px": 0.1,
                "Node_HandLeft_Time": t + 100.0,
                "LeftHand_Root_px": 0.3,
                "RightHand_Root_px": -0.3,
                "LeftHand_Status_HandTracked": 1,
                "RightHand_Status_HandTracked": 1,
            }
        ),
    )
    return session_dir


def _motion(bids_root: Path, system: str) -> pd.DataFrame:
    """The raw motion.tsv of sub-01 ses-01 with the column names from channels.tsv."""
    motion_dir = bids_root / "sub-01" / "ses-01" / "motion"
    (channels,) = motion_dir.glob(f"*_tracksys-{system}_channels.tsv")
    (motion,) = motion_dir.glob(f"*_tracksys-{system}_motion.tsv")
    names = pd.read_csv(channels, sep="\t")["name"].tolist()
    return pd.read_csv(motion, sep="\t", header=None, names=names, na_values="n/a")


def test_zero_clock_values_between_valid_values_are_warned_and_written_as_n_a(
    tmp_path, minimal_config_dict, caplog
):
    data_dir = tmp_path / "sessions"
    session_dir = _write_long_clocked_session(data_dir, "sess_a", session_id="A")
    _edit_continuous(
        session_dir,
        lambda df: df.assign(
            timeSinceStartup=df["timeSinceStartup"].where(~df.index.isin(range(10, 13)), 0.0),
            Node_HandLeft_Time=df["Node_HandLeft_Time"].where(~df.index.isin([20, 21]), 0.0),
        ),
    )
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    bids_root = tmp_path / "bids_out"
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[{"source_dir": "sess_a", "subject_id": "01", "session_label": "01"}],
    )
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert result.exit_code == 0
    (path,) = session_dir.glob("*_ContinuousData.csv")
    assert (
        f"{path}: 3 row(s) have a zero timeSinceStartup between valid values; their global "
        "latency is n/a"
    ) in caplog.text
    for system in ("Head", "Hands"):
        motion = _motion(bids_root, system)
        assert motion["latency_global"].isna().tolist() == [i in (10, 11, 12) for i in range(30)]
        assert (motion["latency_global"].dropna() >= 0).all()
    hands = _motion(bids_root, "Hands")
    assert hands["latency"].isna().tolist() == [i in (20, 21) for i in range(30)]
    assert (hands["latency"].dropna() >= 0).all()


# ── Skipped-session cleanup, follow-ups of the review (16 Oct) ───────


def _edit_config(cfg: Path, edit) -> None:
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    edit(data)
    cfg.write_text(yaml.dump(data), encoding="utf-8")


def test_a_rerun_that_keeps_the_sourcedata_copy_names_every_stale_folder(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_config(cfg, lambda c: c["output"].update(overwrite=False))
    bids_root = tmp_path / "bids_out"
    run(str(cfg))  # a complete first run
    sourcedata = bids_root / "sourcedata" / "sub-01" / "ses-01"
    copied = _files(sourcedata)
    _zero_hands_clock(data_dir / "sess_a")  # now skipped at the split
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert _files(sourcedata) == copied  # the copy was not redone (overwrite: false)
    assert "may now hold partial files" not in caplog.text
    stale = [r for r in caplog.records if "holds output of an earlier run" in r.getMessage()]
    assert [r.levelname for r in stale] == ["WARNING"] * 3
    assert [r.getMessage() for r in stale] == [
        f"{folder} holds output of an earlier run for a session that was skipped now (sess_a); "
        "remove it or fix the session and rerun."
        for folder in (
            sourcedata,
            bids_root / "sub-01" / "ses-01",
            bids_root / "derivatives" / "resxr" / "sub-01" / "ses-01",
        )
    ]
    assert "Removed the output" not in caplog.text


def test_a_rerun_with_a_session_now_unreadable_names_its_stale_folders(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    bids_root = tmp_path / "bids_out"
    run(str(cfg))
    (data_dir / "sess_a" / "session_metadata.json").unlink()  # skipped while loading now
    caplog.clear()
    run(str(cfg))
    stale = [r.getMessage() for r in caplog.records if "earlier run" in r.getMessage()]
    assert len(stale) == 3
    assert stale[0].startswith(f"{bids_root / 'sourcedata' / 'sub-01' / 'ses-01'} holds output")


def test_every_session_skipped_leaves_no_empty_output_folders(
    tmp_path, minimal_config_dict, monkeypatch
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_config(cfg, lambda c: c["report"].update(enabled=True))

    def fail_every_report(self, session, output_path, *args, **kwargs):
        output_path.write_text("partial", encoding="utf-8")
        raise TypeError("must be real number, not str")

    monkeypatch.setattr(pipeline_module.ReportGenerator, "generate", fail_every_report)
    result = run(str(cfg))
    assert result.exit_code == 1
    assert list((tmp_path / "bids_out").iterdir()) == []  # no sourcedata/ or derivatives/


def test_output_folders_that_existed_before_the_run_are_kept_when_empty(
    tmp_path, minimal_config_dict
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    for name in ("sess_a", "sess_b"):
        _zero_hands_clock(data_dir / name)
    bids_root = tmp_path / "bids_out"
    (bids_root / "sourcedata").mkdir(parents=True)
    (bids_root / "derivatives" / "resxr").mkdir(parents=True)
    run(str(cfg))
    assert sorted(p.relative_to(bids_root).as_posix() for p in bids_root.rglob("*")) == [
        "derivatives",
        "derivatives/resxr",
        "sourcedata",
    ]


def test_a_filesystem_error_while_removing_is_a_warning_and_the_run_continues(
    tmp_path, minimal_config_dict, caplog, monkeypatch
):
    data_dir = tmp_path / "sessions"
    _write_clocked_session_dir(data_dir, "sess_a", session_id="A")
    _write_clocked_session_dir(data_dir, "sess_b", session_id="B")
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    bids_root = tmp_path / "bids_out"
    cfg = _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=bids_root,
        session_mappings=[
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01"},
            {"source_dir": "sess_b", "subject_id": "02", "session_label": "01"},
        ],
        report=True,
    )
    _fail_report_of_session_a(monkeypatch)
    unreadable = bids_root / "sub-01"
    iterdir = Path.iterdir

    def iterdir_or_fail(self):
        if self == unreadable:
            raise PermissionError(13, "Access is denied", str(self))
        return iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir_or_fail)
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    (warning,) = [r for r in caplog.records if r.getMessage().startswith("Could not check")]
    assert warning.levelname == "WARNING"
    error = PermissionError(13, "Access is denied", str(unreadable))
    assert warning.getMessage() == (
        f"Could not check {unreadable} of the skipped session sess_a: {error}"
    )
    assert "Run summary: 2 session(s) found, 1 processed, 1 skipped." in caplog.text
    assert not (bids_root / "sub-01" / "ses-01").exists()  # the rest was removed


def test_an_output_folder_that_cannot_be_checked_is_kept_and_the_run_continues(
    tmp_path, minimal_config_dict, caplog, monkeypatch
):
    _, cfg = _two_session_config(tmp_path, minimal_config_dict)
    unreadable = tmp_path / "bids_out" / "sub-01" / "ses-01"
    exists = Path.exists

    def exists_or_fail(self, *args, **kwargs):
        if self == unreadable:
            raise PermissionError(13, "Access is denied", str(self))
        return exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists_or_fail)
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [True, True]
    (warning,) = [r for r in caplog.records if r.getMessage().startswith("Could not check")]
    assert warning.levelname == "WARNING"
    error = PermissionError(13, "Access is denied", str(unreadable))
    assert warning.getMessage() == (
        f"Could not check {unreadable} of session sess_a: {error}; it is kept"
    )


def _make_read_only(folder: Path) -> None:
    for path in folder.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)


def test_read_only_input_files_are_removed_from_sourcedata_of_a_skipped_session(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _zero_hands_clock(data_dir / "sess_a")
    _make_read_only(data_dir / "sess_a")  # copytree copies the read-only attribute
    bids_root = tmp_path / "bids_out"
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert "Could not remove" not in caplog.text
    assert _session_outputs(bids_root, "sub-01") == []


def test_a_rerun_with_overwrite_copies_over_a_read_only_sourcedata_copy(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _make_read_only(data_dir / "sess_a")
    assert run(str(cfg)).exit_code == 0  # output.overwrite is true in this config
    caplog.clear()
    result = run(str(cfg))
    assert result.exit_code == 0
    assert "Access is denied" not in caplog.text and "Permission denied" not in caplog.text
    sourcedata = tmp_path / "bids_out" / "sourcedata" / "sub-01" / "ses-01"
    assert _files(sourcedata) == _files(data_dir / "sess_a")


def test_overwrite_copies_the_input_over_the_sourcedata_copy_and_keeps_other_files(
    tmp_path, minimal_config_dict
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    assert run(str(cfg)).exit_code == 0  # output.overwrite is true in this config
    sourcedata = tmp_path / "bids_out" / "sourcedata" / "sub-01" / "ses-01"
    (sourcedata / "not_in_the_input.txt").write_text("earlier", encoding="utf-8")
    assert run(str(cfg)).exit_code == 0
    assert _files(sourcedata) == {**_files(data_dir / "sess_a"), "not_in_the_input.txt": b"earlier"}


def test_a_direct_call_removes_the_output_of_a_failing_session(tmp_path, minimal_config_dict):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _zero_hands_clock(data_dir / "sess_a")
    config = PipelineConfig.from_yaml(cfg)
    bids = BIDSLayout(config.output.bids_root, config.output.task_name)
    mapping = SessionMapping(source_dir="sess_a", subject_id="01", session_label="01")
    with pytest.raises(DataLoadError, match="Stream Hands: no non-zero timestamp found"):
        process_session_from_mapping(mapping, config, bids, {})
    assert _session_outputs(tmp_path / "bids_out", "sub-01") == []


def test_keyboard_interrupt_removes_the_output_of_the_interrupted_session(
    tmp_path, minimal_config_dict, caplog, monkeypatch
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_config(cfg, lambda c: c["report"].update(enabled=True))

    def interrupt(self, session, output_path, *args, **kwargs):
        output_path.write_text("partial", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module.ReportGenerator, "generate", interrupt)
    with caplog.at_level(logging.INFO, logger="resxr"), pytest.raises(KeyboardInterrupt):
        run(str(cfg))
    assert list((tmp_path / "bids_out").iterdir()) == []
    assert (
        "Removed the output written for interrupted session sess_a: derivatives, sourcedata, sub-01"
    ) in caplog.text


def test_two_sessions_sharing_a_report_path_are_named(tmp_path, minimal_config_dict, caplog):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_metadata(data_dir / "sess_b", lambda m: m.update(session_id="A"))
    reports = tmp_path / "reports"
    _edit_config(cfg, lambda c: c["report"].update(enabled=True, output_dir=str(reports)))
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert result.exit_code == 0
    (warning,) = [r for r in caplog.records if "written earlier in this run" in r.getMessage()]
    assert warning.levelname == "WARNING"
    assert warning.getMessage() == (
        f"The report {reports / 'A_report.html'} was written earlier in this run for sess_a and "
        "is overwritten now by the report of sess_b: the two sessions have the same "
        "session_id, which names the report in report.output_dir"
    )
    assert [p.name for p in reports.iterdir()] == ["A_report.html"]


def test_sessions_read_from_their_own_sourcedata_copies(tmp_path, minimal_config_dict, caplog):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    bids_root = tmp_path / "bids_out"
    run(str(cfg))  # first run: sourcedata/sub-0N/ses-01 hold the copies of sess_a and sess_b
    copies = [bids_root / "sourcedata" / sub / "ses-01" for sub in ("sub-01", "sub-02")]
    _zero_hands_clock(copies[1])  # sub-02 is now skipped at the split
    before = [_files(copy) for copy in copies]

    def reprocess_from_sourcedata(c):  # output.overwrite is true in this config
        c["input"]["data_dir"] = str(bids_root / "sourcedata")
        c["session_mappings"] = [
            {"source_dir": f"sub-{n}/ses-01", "subject_id": n, "session_label": "01"}
            for n in ("01", "02")
        ]

    _edit_config(cfg, reprocess_from_sourcedata)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [True, False]
    assert f"sourcedata is the input folder; nothing copied ({copies[0]})" in caplog.text
    assert [_files(copy) for copy in copies] == before
    stale = [r.getMessage() for r in caplog.records if "earlier run" in r.getMessage()]
    assert stale == [  # none for the skipped session's input folder, sourcedata/sub-02/ses-01
        f"{folder} holds output of an earlier run for a session that was skipped now "
        "(sub-02/ses-01); remove it or fix the session and rerun."
        for folder in (
            bids_root / "sub-02" / "ses-01",
            bids_root / "derivatives" / "resxr" / "sub-02" / "ses-01",
        )
    ]


@pytest.mark.parametrize(
    "face_text",
    [
        "timeSinceStartup;Face_Time;Eyes_Closed_L\n1.0;50.0;0.1\n1.1;50.1;0.2\n",
        "Face_Status,Eyes_Closed_L\ntrue,0.1\ntrue,0.2\n",
    ],
    ids=["semicolons", "no_clock_column"],
)
def test_a_face_file_without_a_clock_column_names_the_expected_format(
    tmp_path, minimal_config_dict, caplog, face_text
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_config(
        cfg,
        lambda c: c["preprocessing"]["alternate_time_columns"].update(Face="Face_Time"),
    )
    session_dir = data_dir / "sess_a"
    _edit_metadata(session_dir, lambda m: m.update(face_enabled=True))
    face = session_dir / "A_FaceExpressionData.csv"
    face.write_text(face_text, encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="resxr"):
        result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]  # still skipped
    assert result.skipped[0].reason == (
        f"No timestamp column found in {face}: the file must be comma-separated, with a header "
        "row that includes timeSinceStartup or Face_Time"
    )
    assert "clock column 'Face_Time'" not in caplog.text
    assert "missing 'timestamp' column" not in caplog.text


def test_a_face_file_that_pandas_cannot_tokenize_names_the_expected_format(
    tmp_path, minimal_config_dict, caplog
):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    face = data_dir / "sess_a" / "A_FaceExpressionData.csv"
    face.write_text("timeSinceStartup;Eyes_Closed_L\n1,0;0,1\n1,1;0,2;3,4,5\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="resxr"):
        run(str(cfg))
    (warning,) = [r for r in caplog.records if r.getMessage().startswith("Could not load face")]
    assert warning.getMessage().startswith(f"Could not load face data: Failed to load {face}: ")
    assert warning.getMessage().endswith(
        ". Check that the file is comma-separated with '.' as the decimal separator, and that "
        "every row has the header's number of fields"
    )


def test_a_text_fixed_delta_time_no_longer_stops_the_report(tmp_path, minimal_config_dict):
    data_dir, cfg = _two_session_config(tmp_path, minimal_config_dict)
    _edit_config(cfg, lambda c: c["report"].update(enabled=True))
    _edit_metadata(data_dir / "sess_a", lambda m: m.update(fixedDeltaTime="0.02"))
    assert run(str(cfg)).exit_code == 0
    report = (tmp_path / "bids_out" / "sub-01" / "ses-01" / "A_report.html").read_text("utf-8")
    assert "<tr><th>Fixed delta time</th><td>0.0200 s</td></tr>" in report


# ── participants.tsv: one row per subject (D13) ──────────────────────


def _subject_sessions_config(tmp_path, minimal_config_dict, mappings: list[dict]) -> Path:
    """One clocked session folder per mapping (sess_a has session_id A, ...), and the config."""
    data_dir = tmp_path / "sessions"
    for mapping in mappings:
        name = mapping["source_dir"]
        _write_clocked_session_dir(data_dir, name, session_id=name.removeprefix("sess_").upper())
    minimal_config_dict["preprocessing"]["alternate_time_columns"] = {
        "Head": "Node_Head_Time",
        "Hands": "Node_HandLeft_Time",
    }
    return _write_config(
        tmp_path,
        minimal_config_dict,
        data_dir=data_dir,
        bids_root=tmp_path / "bids_out",
        session_mappings=mappings,
    )


def test_a_subject_with_two_processed_sessions_has_one_participants_row(
    tmp_path, minimal_config_dict
):
    demographics = {"age": "25", "sex": "F", "handedness": "R"}
    cfg = _subject_sessions_config(
        tmp_path,
        minimal_config_dict,
        [
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01", **demographics},
            {"source_dir": "sess_b", "subject_id": "02", "session_label": "01"},
            {"source_dir": "sess_c", "subject_id": "01", "session_label": "02", **demographics},
        ],
    )
    result = run(str(cfg))
    assert result.exit_code == 0
    assert len(result.processed) == 3
    bids_root = tmp_path / "bids_out"
    assert (bids_root / "participants.tsv").read_text(encoding="utf-8") == (
        "participant_id\tage\tsex\thandedness\nsub-01\t25\tF\tR\nsub-02\tn/a\tn/a\tn/a\n"
    )
    for session in ("ses-01", "ses-02"):  # both sessions of sub-01 are exported
        assert list((bids_root / "sub-01" / session / "motion").glob("*_motion.tsv"))


def test_a_subject_whose_first_session_is_skipped_is_listed_once(tmp_path, minimal_config_dict):
    cfg = _subject_sessions_config(
        tmp_path,
        minimal_config_dict,
        [
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01", "sex": "F"},
            {"source_dir": "sess_b", "subject_id": "01", "session_label": "02", "sex": "F"},
        ],
    )
    _zero_hands_clock(tmp_path / "sessions" / "sess_a")  # skipped at the split
    result = run(str(cfg))
    assert [s.processed for s in result.sessions] == [False, True]
    assert (tmp_path / "bids_out" / "participants.tsv").read_text(encoding="utf-8") == (
        "participant_id\tage\tsex\thandedness\nsub-01\tn/a\tF\tn/a\n"
    )


def test_different_demographics_for_one_subject_stop_the_run_before_any_session(
    tmp_path, minimal_config_dict, caplog
):
    cfg = _subject_sessions_config(
        tmp_path,
        minimal_config_dict,
        [
            {"source_dir": "sess_a", "subject_id": "01", "session_label": "01", "age": "25"},
            {"source_dir": "sess_b", "subject_id": "01", "session_label": "02", "age": "26"},
        ],
    )
    message = (
        f"Configuration error in {cfg}: session_mappings gives sub-01 different age values: "
        "'25' (sess_a), '26' (sess_b); participants.tsv has one row per subject, so every "
        "mapping of sub-01 must give the same age, or none may give it"
    )
    for dry_run in ([], ["--dry-run"]):
        caplog.clear()
        assert main(["-c", str(cfg), *dry_run]) == 1
        assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == [message]
        assert "Processing session" not in caplog.text
        assert "Configuration valid" not in caplog.text
    assert not (tmp_path / "bids_out").exists()
