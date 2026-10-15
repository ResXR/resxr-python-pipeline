"""
High-level orchestration for the ResXR pipeline.

This module is the single place that describes the end-to-end flow::

    load data → split → validate → preprocess → export BIDS → report

BIDS output (``write_bids_output``):

    Before writing each stream, ``prepare_motion_data`` is called to
    convert internal time columns (``timestamp``, ``timeSinceStartup``)
    into BIDS-compliant LATENCY channels (``latency``, ``latency_global``)
    and strip the originals.  The resulting DataFrame is passed to
    ``write_motion_tsv``, ``generate_channels_tsv``, and
    ``generate_motion_json`` so all three outputs describe the same columns.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .bids.channels import TRACKING_SPACE_FRAME, detect_hand_joint_frames, generate_channels_tsv
from .bids.events_merge import generate_events_sidecar, merge_events
from .bids.layout import BIDSLayout
from .bids.metadata import (
    generate_channels_json,
    generate_dataset_description,
    generate_derivative_description,
    generate_motion_json,
    generate_participants_json,
)
from .core.config import PipelineConfig, SessionMapping, ValidationConfig
from .core.constants import TrackingSystem
from .core.exceptions import ConfigurationError, DataLoadError, ResXRError
from .core.logger import get_logger
from .core.session import CheckNotRun, Session
from .io.readers import discover_sessions, load_session
from .io.splitter import split_continuous_data, systems_off_on_purpose
from .io.writers import (
    copy_sourcedata,
    write_bids_events,
    write_channels_tsv,
    write_json,
    write_motion_tsv,
    write_participants_tsv,
    write_scans_tsv,
)
from .preprocessing.stream_preprocessing import prepare_motion_data, preprocess_stream
from .validation import check_registry
from .visualization.report import ReportGenerator

logger = get_logger(__name__)


def validate_config(config: PipelineConfig) -> None:
    """Reject settings that would otherwise be ignored without a message.

    Raises ConfigurationError naming every problem found: check names that no
    registered check has (in ``validation.enabled_checks`` or
    ``preprocessing.masking_checks``), tracking-system names that do not exist
    (keys of ``systems``, ``sampling_frequencies``, ``system_descriptions`` and
    ``preprocessing.alternate_time_columns``), sampling rates that are not
    positive, keys under ``validation.settings`` that no registered check
    declares in its ``settings_keys`` (``check_column_groups`` is read by
    ``ValidationConfig.get_column_groups``), and, when
    ``preprocessing.apply_quality_masking`` is true, ``masking_checks`` names
    that are not in ``validation.enabled_checks``.
    """
    problems = []
    known_checks = check_registry.available_checks
    for setting, names in (
        ("validation.enabled_checks", config.validation.enabled_checks),
        ("preprocessing.masking_checks", config.preprocessing.masking_checks or []),
    ):
        unknown = [name for name in names if name not in known_checks]
        if unknown:
            problems.append(
                f"{setting} names unknown check(s) {unknown}; valid check names: "
                f"{sorted(known_checks)}"
            )

    system_names = [system.value for system in TrackingSystem]
    for setting, keys in (
        ("systems", config.systems),
        ("sampling_frequencies", config.sampling_frequencies),
        ("system_descriptions", config.system_descriptions),
        ("preprocessing.alternate_time_columns", config.preprocessing.alternate_time_columns),
    ):
        unknown = [key for key in keys if key not in system_names]
        if unknown:
            problems.append(
                f"{setting} names unknown tracking system(s) {unknown}; valid names: {system_names}"
            )

    not_positive = {name: rate for name, rate in config.sampling_frequencies.items() if rate <= 0}
    if not_positive:
        problems.append(f"sampling_frequencies must be positive; got {not_positive}")

    known_settings = sorted(set(check_registry.settings_keys) | set(ValidationConfig.SETTINGS_KEYS))
    unknown_settings = [key for key in config.validation.settings if key not in known_settings]
    if unknown_settings:
        problems.append(
            f"validation.settings has unknown key(s) {unknown_settings}; keys the registered "
            f"checks read: {known_settings}"
        )

    if config.preprocessing.apply_quality_masking and config.preprocessing.masking_checks:
        enabled = set(config.validation.enabled_checks)
        not_enabled = [  # unknown names are reported above
            name
            for name in dict.fromkeys(config.preprocessing.masking_checks)
            if name in known_checks and name not in enabled
        ]
        if not_enabled:
            problems.append(
                f"preprocessing.masking_checks names check(s) {not_enabled} that are not in "
                "validation.enabled_checks; they never run, so nothing would be masked for "
                "them. Add them to enabled_checks or remove them from masking_checks."
            )

    if problems:
        raise ConfigurationError("; ".join(problems))


@dataclass
class SessionOutcome:
    """What happened to one session of a run."""

    source_dir: str
    subject_id: str
    session_label: str
    processed: bool
    reason: str = ""  # why a skipped session was skipped
    warnings: int = 0  # WARNING messages logged while processing the session
    errors: int = 0  # ERROR messages logged while processing the session
    checks_not_run: list[CheckNotRun] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of :func:`run`: one entry per session found."""

    output_dir: Path
    sessions: list[SessionOutcome] = field(default_factory=list)

    @property
    def processed(self) -> list[SessionOutcome]:
        return [s for s in self.sessions if s.processed]

    @property
    def skipped(self) -> list[SessionOutcome]:
        return [s for s in self.sessions if not s.processed]

    @property
    def exit_code(self) -> int:
        """0 when every session found was processed; 1 when one was skipped or none found."""
        return 0 if self.sessions and not self.skipped else 1


class _MessageCounter(logging.Handler):
    """Counts the WARNING and ERROR records that reach the ``resxr`` logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warnings = 0
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.errors += 1
        else:
            self.warnings += 1


def _report_path(session: Session, config: PipelineConfig, bids: BIDSLayout) -> Path:
    """Where the session's HTML report is written."""
    report_dir = config.report.output_dir or bids.get_session_dir(session)
    return Path(report_dir) / f"{session.session_id}_report.html"


class _SessionOutputs:
    """The output one session writes, so that a skipped session leaves none behind.

    ``start`` records, before the session writes anything, which of its output
    targets already exist: its sourcedata, raw and derivative session folders
    (``ses-*``, with the ``sub-*`` folders above them) and its report file.
    Each write step marks its target with ``writing``. ``remove`` deletes the
    marked targets that the session created, then the ``sub-*`` folders that
    it created and left empty; a marked target that existed before is kept,
    with a warning. Nothing outside the BIDS root and the report directory is
    deleted, nor any path that is, holds or lies inside a session's input
    folder, nor the data directory or a folder above it.
    """

    def __init__(self, config: PipelineConfig, bids: BIDSLayout, source_dirs: list[Path]):
        self.bids_root = bids.bids_root
        self.roots = [bids.bids_root.resolve()]
        if config.report.output_dir is not None:
            self.roots.append(Path(config.report.output_dir).resolve())
        self.data_dir = Path(config.input.data_dir).resolve()
        self.source_dirs = [Path(d).resolve() for d in source_dirs]
        self.existed: dict[Path, bool] = {}  # target or its sub-* folder -> existed at start
        self.targets: list[Path] = []
        self.marked: list[Path] = []

    def start(self, session: Session, config: PipelineConfig, bids: BIDSLayout) -> None:
        """Record which output targets of the session exist before it writes any."""
        folders = [
            bids.sourcedata_session_dir(session),
            bids.get_session_dir(session),
            bids.get_session_dir(session, derivative=True),
        ]
        report = [_report_path(session, config, bids)] if config.report.enabled else []
        self.targets = folders + report
        for target in self.targets:
            self.existed[target] = target.exists()
        for folder in folders:
            self.existed.setdefault(folder.parent, folder.parent.exists())

    def writing(self, target: Path) -> None:
        """Mark a target that the session is about to write."""
        if target in self.existed and target not in self.marked:
            self.marked.append(target)

    def remove(self, source_dir: str) -> None:
        """Delete what the skipped session created; warn about targets that existed."""
        removed: list[Path] = []
        for target in self.marked:
            if self.existed[target]:
                logger.warning(
                    f"{target} existed before this run and may now hold partial files from "
                    f"the skipped session {source_dir}; remove it or rerun."
                )
            elif target.exists():
                action = shutil.rmtree if target.is_dir() else Path.unlink
                if self._remove(target, action):
                    removed.append(target)
        above = [p for p, existed in self.existed.items() if not existed and p not in self.targets]
        for folder in sorted(above, key=lambda p: len(p.parts), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()) and self._remove(folder, Path.rmdir):
                removed.append(folder)
        shown = [p for p in removed if not any(q in p.parents for q in removed)]
        if shown:
            names = ", ".join(sorted(self._shown(p) for p in shown))
            logger.info(f"Removed the output written for skipped session {source_dir}: {names}")

    def _deletable(self, path: Path) -> bool:
        p = path.resolve()
        inside_output = any(root in p.parents for root in self.roots)
        input_data = p == self.data_dir or p in self.data_dir.parents
        input_data = input_data or any(
            p == d or d in p.parents or p in d.parents for d in self.source_dirs
        )
        return inside_output and not input_data

    def _remove(self, path: Path, action) -> bool:
        if not self._deletable(path):
            logger.warning(f"Not removed: {path} is outside the output folders or holds input data")
            return False
        try:
            action(path)
        except OSError as e:
            logger.warning(f"Could not remove {path} of a skipped session: {e}")
            return False
        return True

    def _shown(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.bids_root))
        except ValueError:
            return str(path)


def run(config_path: str = "config/pipeline_config.yaml") -> RunResult:
    """
    Run the complete ResXR pipeline.

    A session that cannot be processed is logged as an error and skipped; the
    run continues with the others and ends with a summary of every session.
    An unexpected error (not a ResXRError) is handled the same way, with its
    traceback at DEBUG; KeyboardInterrupt still stops the run. The output that
    a skipped session had written (sourcedata copy, raw and derivative session
    files, report) is removed; folders that existed before are kept, with a
    warning.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Returns
    -------
    RunResult
        The processed and skipped sessions; ``exit_code`` is 0 only when every
        session found was processed. Warning counts include only messages at or
        above the ``resxr`` logger's level (none in the CLI's ``--quiet`` mode).

    Raises
    ------
    ConfigurationError
        If the configuration is invalid (before any session is read).
    """
    # Load and check the configuration before any session is processed
    config = PipelineConfig.from_yaml(Path(config_path))
    validate_config(config)

    # Setup logging
    logger.info(f"Starting ResXR pipeline with config: {config_path}")

    # Initialize BIDS layout
    bids = BIDSLayout(config.output.bids_root, config.output.task_name)

    # Get enabled systems from config
    enabled_systems = {name: cfg.enabled for name, cfg in config.systems.items()}

    if config.session_mappings:
        mappings = list(config.session_mappings)
    else:
        # Auto-discover sessions
        logger.info("No session mappings provided, discovering sessions...")
        session_dirs = discover_sessions(config.input)
        mappings = [
            SessionMapping(
                source_dir=str(session_dir), subject_id=f"{i + 1:02d}", session_label="01"
            )
            for i, session_dir in enumerate(session_dirs)
        ]
        if not mappings:
            _log_no_sessions_found(config)

    result = RunResult(output_dir=config.output.bids_root)
    processed_subjects = []
    resxr_logger = logging.getLogger("resxr")
    source_dirs = [_source_dir(mapping, config) for mapping in mappings]
    for mapping in mappings:
        outputs = _SessionOutputs(config, bids, source_dirs)
        outcome = SessionOutcome(
            source_dir=mapping.source_dir,
            subject_id=mapping.subject_id,
            session_label=mapping.session_label,
            processed=False,
        )
        counter = _MessageCounter()
        resxr_logger.addHandler(counter)
        unexpected: Exception | None = None
        try:
            session = process_session_from_mapping(
                mapping, config, bids, enabled_systems, outputs=outputs
            )
            outcome.processed = True
            outcome.checks_not_run = list(session.checks_not_run)
            processed_subjects.append(
                {
                    "participant_id": f"sub-{session.subject_id}",
                    "age": mapping.age if mapping.age is not None else "n/a",
                    "sex": mapping.sex if mapping.sex is not None else "n/a",
                    "handedness": mapping.handedness if mapping.handedness is not None else "n/a",
                }
            )
        except ResXRError as e:
            outcome.reason = str(e)
        except Exception as e:  # not foreseen by the code: skip this session only
            outcome.reason = f"unexpected error: {type(e).__name__}: {e}"
            unexpected = e
        finally:
            resxr_logger.removeHandler(counter)
        outcome.warnings, outcome.errors = counter.warnings, counter.errors
        if not outcome.processed:
            logger.error(f"Failed to process {mapping.source_dir}: {outcome.reason}")
            if unexpected is not None:
                logger.debug("Traceback of the unexpected error:", exc_info=unexpected)
            outputs.remove(mapping.source_dir)
        result.sessions.append(outcome)

    # Write dataset-level files only if at least one session succeeded
    if processed_subjects:
        write_dataset_files(config, bids, processed_subjects)

    log_run_summary(result)
    if result.processed:
        logger.info(f"Pipeline complete. Output at: {config.output.bids_root}")
    return result


def _log_no_sessions_found(config: PipelineConfig) -> None:
    """ERROR for auto-discovery that found nothing, naming nested candidates."""
    data_dir = config.input.data_dir
    pattern = config.input.metadata_pattern
    nested = sorted({p.parent for p in data_dir.rglob(pattern)})
    hint = ""
    if nested:
        shown = ", ".join(str(p) for p in nested[:3])
        more = f" and {len(nested) - 3} more" if len(nested) > 3 else ""
        hint = (
            f" {len(nested)} deeper folder(s) contain one ({shown}{more}); "
            "list them under session_mappings."
        )
    logger.error(
        f"No sessions found under {data_dir}: without session_mappings, only direct "
        f"subfolders that contain a file matching '{pattern}' are sessions.{hint}"
    )


def log_run_summary(result: RunResult) -> None:
    """Log one line per session: processed (warnings, checks not run) or skipped (reason).

    The level is ERROR when no session was processed, WARNING when a session was
    skipped or a check did not run, and INFO otherwise.
    """
    processed, skipped = result.processed, result.skipped
    if not processed:
        level = logging.ERROR
    elif skipped or any(s.checks_not_run for s in processed):
        level = logging.WARNING
    else:
        level = logging.INFO

    logger.log(
        level,
        f"Run summary: {len(result.sessions)} session(s) found, {len(processed)} processed, "
        f"{len(skipped)} skipped.",
    )
    for s in result.sessions:
        target = f"sub-{s.subject_id} ses-{s.session_label}"
        if not s.processed:
            logger.log(level, f"  skipped {s.source_dir} ({target}): {s.reason}")
            continue
        line = (
            f"  processed {s.source_dir} ({target}): {s.warnings} warning(s), {s.errors} error(s)"
        )
        if s.checks_not_run:
            not_run = "; ".join(f"{c.check} on {c.stream}: {c.reason}" for c in s.checks_not_run)
            line += f"; checks not run: {not_run}"
        logger.log(level, line)


def _source_dir(mapping: SessionMapping, config: PipelineConfig) -> Path:
    """The session's input folder (``source_dir`` relative to ``input.data_dir``)."""
    source_dir = Path(mapping.source_dir)
    return source_dir if source_dir.is_absolute() else config.input.data_dir / source_dir


def process_session_from_mapping(
    mapping: SessionMapping,
    config: PipelineConfig,
    bids: BIDSLayout,
    enabled_systems: dict[str, bool],
    outputs: _SessionOutputs | None = None,
) -> Session:
    """
    Process a single session based on a mapping.

    Parameters
    ----------
    mapping : SessionMapping
        Session mapping with source dir and BIDS identifiers
    config : PipelineConfig
        Pipeline configuration
    bids : BIDSLayout
        BIDS layout manager
    enabled_systems : Dict[str, bool]
        Which tracking systems are enabled
    outputs : _SessionOutputs, optional
        Records what the session writes, so that ``run`` can remove it when the
        session is skipped

    Returns
    -------
    Session
        The processed session

    Raises
    ------
    ResXRError
        When the session cannot be processed; the caller logs it and skips it.
    """
    # Resolve session directory
    source_dir = _source_dir(mapping, config)

    logger.info(f"Processing session: {source_dir}")

    # Load session
    session = load_session(Path(source_dir), config.input)
    session.subject_id = mapping.subject_id
    session.session_label = mapping.session_label
    if outputs is None:  # direct callers: track the session on its own
        outputs = _SessionOutputs(config, bids, [source_dir])
    outputs.start(session, config, bids)

    # Copy the raw session verbatim into sourcedata/ before any transformation.
    outputs.writing(bids.sourcedata_session_dir(session))
    copy_sourcedata(
        Path(source_dir),
        bids.sourcedata_session_dir(session),
        overwrite=config.output.overwrite,
    )

    # Split into tracking streams
    session.streams = split_continuous_data(
        session,
        enabled_systems,
        config.sampling_frequencies,
        alternate_time_columns=config.preprocessing.alternate_time_columns,
    )
    logger.info(f"Split into {len(session.streams)} tracking streams")
    session.systems_off = systems_off_on_purpose(session.metadata, enabled_systems)

    if not session.streams:
        if session.raw_continuous_data is None or session.raw_continuous_data.empty:
            why = "the continuous data file has no data rows"
        else:
            why = "no enabled tracking system has data columns (see the messages above)"
        raise DataLoadError(f"No tracking stream to export from {source_dir}: {why}")

    session.initialize_motion_time_zero()

    hands = session.get_stream(TrackingSystem.HANDS)
    if hands is not None:
        session.hand_joint_frames = detect_hand_joint_frames(hands.data, session.session_id)
    if session.hand_joint_frames:
        frames = ", ".join(f"{side} {frame}" for side, frame in session.hand_joint_frames.items())
        logger.info(f"Hand joint frame in {session.session_id}: {frames}")

    check_registry.clear_failed_checks()
    for system, stream in session.streams.items():
        logger.info(f"Validating {system.value}")
        flags = check_registry.run_all(stream, session, config.validation)
        for flag in flags:
            (session.get_stream(flag.system) or stream).quality_flags.append(flag)
        logger.info(f"  Found {len(flags)} quality flags")

    session.merged_events_data = merge_events(
        session.raw_events_data,
        session.custom_tables_data,
        session.custom_tables,
        time_zero=session.motion_time_zero,
    )

    # Write RAW BIDS output (original data)
    logger.info("Writing RAW BIDS dataset (original)...")
    outputs.writing(bids.get_session_dir(session))
    write_bids_output(session, config, bids, derivative=False)

    # Preprocess (masking)
    for stream in session.streams.values():
        preprocess_stream(
            stream,
            apply_masking=config.preprocessing.apply_quality_masking,
            masking_checks=config.preprocessing.masking_checks,
        )

    # Write DERIVATIVE BIDS output (masked/preprocessed data)
    logger.info("Writing DERIVATIVE BIDS dataset (preprocessed)...")
    outputs.writing(bids.get_session_dir(session, derivative=True))
    write_bids_output(session, config, bids, derivative=True)

    # Generate report
    if config.report.enabled:
        report_path = _report_path(session, config, bids)
        outputs.writing(report_path)
        reporter = ReportGenerator(config.report)
        reporter.generate(session, report_path, config=config)

    return session


def write_bids_output(
    session: Session, config: PipelineConfig, bids: BIDSLayout, derivative: bool = False
) -> None:
    """
    Write all BIDS-compliant output files for a session.

    Parameters
    ----------
    session : Session
        Processed session
    config : PipelineConfig
        Pipeline configuration
    bids : BIDSLayout
        BIDS layout manager
    """
    # Create directory structure
    bids.create_structure(session, derivative=derivative)

    # Track files for scans.tsv
    scans_entries = []

    # Write per-system files
    for system, stream in session.streams.items():
        # Use clean_data for derivative dataset, raw data for raw dataset
        data = stream.get_output_data() if derivative else stream.data
        if data.empty:
            logger.warning(f"Skipping empty stream: {system.value}")
            continue

        # Prepare for BIDS output (add LATENCY channels, strip internal time cols)
        prepared = prepare_motion_data(data, global_onset=session.motion_time_zero)

        # Write motion.tsv (no header)
        motion_path = bids.get_motion_file(session, system, "tsv", derivative=derivative)
        write_motion_tsv(prepared, motion_path, config.bids.missing_values)

        # Write motion.json
        json_path = bids.get_motion_file(session, system, "json", derivative=derivative)
        metadata = generate_motion_json(
            stream,
            session,
            config.output.task_name,
            config.device,
            config.bids,
            config.system_descriptions,
            prepared_data=prepared,
        )
        write_json(metadata, json_path)

        # Write channels.tsv
        channels_path = bids.get_channels_file(session, system, "tsv", derivative=derivative)
        channels_df = generate_channels_tsv(
            prepared, stream.sampling_frequency, session.hand_joint_frames
        )
        write_channels_tsv(channels_df, channels_path)

        # Write channels.json
        channels_json_path = bids.get_channels_file(session, system, "json", derivative=derivative)
        channels_meta = generate_channels_json(
            config.bids,
            include_tracking_space=bool(
                (channels_df["reference_frame"] == TRACKING_SPACE_FRAME).any()
            ),
        )
        write_json(channels_meta, channels_json_path)

        # Add to scans list
        relative_path = f"motion/{motion_path.name}"
        acq_time = "n/a"
        if session.metadata.utc_start:
            acq_time = session.metadata.utc_start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        scans_entries.append(
            {
                "filename": relative_path,
                "acq_time": acq_time,
            }
        )

        logger.info(f"Wrote {system.value}: {motion_path.name}")

    # Write scans.tsv
    if scans_entries:
        scans_path = bids.get_scans_file(session, derivative=derivative)
        scans_df = pd.DataFrame(scans_entries)
        write_scans_tsv(scans_df, scans_path)

    # Merged wide events at the SESSION ROOT (not motion/) — raw tier only.
    if (
        not derivative
        and session.merged_events_data is not None
        and not session.merged_events_data.empty
    ):
        events_dir = bids.get_session_dir(session)
        events_dir.mkdir(parents=True, exist_ok=True)
        events_path = events_dir / (
            f"sub-{session.subject_id}_ses-{session.session_label}"
            f"_task-{config.output.task_name}_events.tsv"
        )
        sidecar = generate_events_sidecar(session.merged_events_data, session.custom_tables)
        undescribed = [c for c in session.merged_events_data.columns if c not in sidecar]
        if undescribed:
            logger.warning(
                f"Session '{session.session_id}': events.tsv columns not described in "
                f"events.json: {', '.join(undescribed)} (extra Events CSV columns, or "
                "custom-table columns missing from the CustomTables sidecar)"
            )
        write_bids_events(session.merged_events_data, events_path, sidecar)
        logger.info(f"Wrote events: {events_path.name}")


def write_dataset_files(config: PipelineConfig, bids: BIDSLayout, subjects: list[dict]) -> None:
    """
    Write dataset-level BIDS files.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    bids : BIDSLayout
        BIDS layout manager
    subjects : List[dict]
        List of subject info dictionaries
    """
    # Create root directory
    bids.bids_root.mkdir(parents=True, exist_ok=True)

    # Write README
    readme_path = bids.bids_root / "README"
    if not readme_path.exists() or config.output.overwrite:
        readme_content = (
            config.bids.readme_text
            or f"{config.output.dataset_name}\n\nGenerated by ResXR pipeline.\n"
        )
        readme_path.write_text(readme_content, encoding="utf-8", newline="\n")
        logger.info(f"Wrote: {readme_path.name}")

    # Write dataset_description.json (raw)
    desc_path = bids.get_dataset_description_file()
    if not desc_path.exists() or config.output.overwrite:
        desc = generate_dataset_description(
            config.output.dataset_name,
            config.output.bids_version,
            config.bids,
        )
        write_json(desc, desc_path)
        logger.info(f"Wrote: {desc_path.name}")

    # Write dataset_description.json (derivative)
    deriv_desc_path = bids.get_dataset_description_file(derivative=True)
    deriv_desc_path.parent.mkdir(parents=True, exist_ok=True)
    if not deriv_desc_path.exists() or config.output.overwrite:
        deriv_desc = generate_derivative_description(
            config.output.dataset_name,
            config.output.bids_version,
        )
        write_json(deriv_desc, deriv_desc_path)
        logger.info(f"Wrote derivative: {deriv_desc_path.name}")

    # Write README (derivative)
    deriv_readme_path = deriv_desc_path.parent / "README"
    if not deriv_readme_path.exists() or config.output.overwrite:
        deriv_readme_content = (
            f"{config.output.dataset_name} (preprocessed)\n\nGenerated by ResXR pipeline.\n"
        )
        deriv_readme_path.write_text(deriv_readme_content, encoding="utf-8", newline="\n")
        logger.info(f"Wrote derivative README: {deriv_readme_path.name}")

    # Write participants.tsv
    if subjects:
        participants_path = bids.get_participants_file("tsv")
        participants_df = pd.DataFrame(subjects)
        write_participants_tsv(participants_df, participants_path)
        logger.info(f"Wrote: {participants_path.name}")

        # Write participants.json
        participants_json_path = bids.get_participants_file("json")
        participants_meta = generate_participants_json()
        write_json(participants_meta, participants_json_path)

    # Write .bidsignore
    bidsignore_path = bids.get_bidsignore_file()
    if not bidsignore_path.exists():
        bidsignore_path.write_text("*.html\n*.png\n*.jpg\n.git/\n", encoding="utf-8", newline="\n")
        logger.info(f"Wrote: {bidsignore_path.name}")
