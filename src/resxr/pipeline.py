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

from pathlib import Path

import pandas as pd

from .bids.channels import generate_channels_tsv
from .bids.events_merge import generate_events_sidecar, merge_events
from .bids.layout import BIDSLayout
from .bids.metadata import (
    generate_channels_json,
    generate_dataset_description,
    generate_derivative_description,
    generate_motion_json,
    generate_participants_json,
)
from .core.config import PipelineConfig, SessionMapping
from .core.exceptions import ResXRError
from .core.logger import get_logger
from .core.session import Session
from .io.readers import discover_sessions, load_session
from .io.splitter import split_continuous_data
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


def run(config_path: str = "config/pipeline_config.yaml") -> None:
    """
    Run the complete ResXR pipeline.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    """
    # Load configuration
    config = PipelineConfig.from_yaml(Path(config_path))

    # Setup logging
    logger.info(f"Starting ResXR pipeline with config: {config_path}")

    # Initialize BIDS layout
    bids = BIDSLayout(config.output.bids_root, config.output.task_name)

    # Get enabled systems from config
    enabled_systems = {name: cfg.enabled for name, cfg in config.systems.items()}

    # Process sessions based on mappings
    processed_subjects = []

    if config.session_mappings:
        # Use explicit mappings
        for mapping in config.session_mappings:
            try:
                session = process_session_from_mapping(mapping, config, bids, enabled_systems)
                if session:
                    processed_subjects.append(
                        {
                            "participant_id": f"sub-{session.subject_id}",
                            "age": mapping.age if mapping.age is not None else "n/a",
                            "sex": mapping.sex if mapping.sex is not None else "n/a",
                            "handedness": mapping.handedness
                            if mapping.handedness is not None
                            else "n/a",
                        }
                    )
            except ResXRError as e:
                logger.error(f"Failed to process {mapping.source_dir}: {e}")
    else:
        # Auto-discover sessions
        logger.info("No session mappings provided, discovering sessions...")
        session_dirs = discover_sessions(config.input)

        for i, session_dir in enumerate(session_dirs):
            mapping = SessionMapping(
                source_dir=str(session_dir),
                subject_id=f"{i + 1:02d}",
                session_label="01",
            )
            try:
                session = process_session_from_mapping(mapping, config, bids, enabled_systems)
                if session:
                    processed_subjects.append(
                        {
                            "participant_id": f"sub-{session.subject_id}",
                            "age": mapping.age if mapping.age is not None else "n/a",
                            "sex": mapping.sex if mapping.sex is not None else "n/a",
                            "handedness": mapping.handedness
                            if mapping.handedness is not None
                            else "n/a",
                        }
                    )
            except ResXRError as e:
                logger.error(f"Failed to process {session_dir}: {e}")

    # Write dataset-level files only if at least one session succeeded
    if processed_subjects:
        write_dataset_files(config, bids, processed_subjects)

    logger.info(f"Pipeline complete. Output at: {config.output.bids_root}")


def process_session_from_mapping(
    mapping: SessionMapping,
    config: PipelineConfig,
    bids: BIDSLayout,
    enabled_systems: dict[str, bool],
) -> Session | None:
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

    Returns
    -------
    Optional[Session]
        Processed session or None if failed
    """
    # Resolve session directory
    source_dir = mapping.source_dir
    if not Path(source_dir).is_absolute():
        source_dir = config.input.data_dir / source_dir

    logger.info(f"Processing session: {source_dir}")

    # Load session
    session = load_session(Path(source_dir), config.input)
    session.subject_id = mapping.subject_id
    session.session_label = mapping.session_label

    # Copy the raw session verbatim into sourcedata/ before any transformation.
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

    if not session.streams:
        logger.warning(f"No valid streams found in {source_dir}")
        return None

    check_registry.clear_failed_checks()
    for system, stream in session.streams.items():
        logger.info(f"Validating {system.value}")
        flags = check_registry.run_all(stream, session, config.validation)
        stream.quality_flags = flags
        logger.info(f"  Found {len(flags)} quality flags")

    session.merged_events_data = merge_events(
        session.raw_events_data,
        session.custom_tables_data,
        session.custom_tables
    )

    # Write RAW BIDS output (original data)
    logger.info("Writing RAW BIDS dataset (original)...")
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
    write_bids_output(session, config, bids, derivative=True)

    # Generate report
    if config.report.enabled:
        report_dir = bids.get_session_dir(session)
        report_path = report_dir / f"{session.session_id}_report.html"
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
        prepared = prepare_motion_data(data)

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
        channels_df = generate_channels_tsv(prepared, stream.sampling_frequency)
        write_channels_tsv(channels_df, channels_path)

        # Write channels.json
        channels_json_path = bids.get_channels_file(session, system, "json", derivative=derivative)
        channels_meta = generate_channels_json(config.bids)
        write_json(channels_meta, channels_json_path)

        # Add to scans list
        relative_path = f"motion/{motion_path.name}"
        acq_time = "n/a"
        if session.metadata.utc_start:
            acq_time = session.metadata.utc_start.strftime("%Y-%m-%dT%H:%M:%S.000")

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
        bidsignore_path.write_text("*.html\n*.png\n*.jpg\n.git/\n")
        logger.info(f"Wrote: {bidsignore_path.name}")
