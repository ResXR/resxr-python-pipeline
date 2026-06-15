# Changelog

All notable changes to the ResXR Python pipeline are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.2.0] - 2026-06-15

This release adds experiment-defined custom data tables, a unified BIDS events
timeline, a `sourcedata/` provenance tier, a new clock-dropout validation check,
and engine-agnostic software/OS version metadata. It also ships a refreshed demo
dataset (Binary Choice, Maze Navigation, Museum Viewing).

### ⚠ Breaking Changes

- **Input file patterns changed.** Metadata is now discovered as
  `*SessionMetadata.json` (was `*session_metadata.json`) and events as
  `*_Events.csv` (was `*events*.csv`). Update your recordings or override the
  patterns in `config/pipeline_config.yaml`.
- **New `SessionMetadata` schema.** Session metadata files in the old shape may no
  longer parse; re-export from the current ResXR Unity template or adapt the file.
- **Events use a `name` column** in place of `trial_type`. Downstream code reading
  `trial_type` from `events.tsv` must switch to `name`.
- **BIDS `missing_values` default changed from `NaN` to `n/a`** (the BIDS-compliant
  token). Consumers expecting the literal `NaN` token must be updated.
- **Removed `Session.session_flags`** from the Python API.

### Added

- **Custom data-class tables.** Config-driven discovery of per-session
  `CustomTables/` folders and parsing of the `custom_tables.json` schema
  (`ColumnInfoEntry` / `CustomTableSchema`), with exact-match preference and
  warnings when multiple or missing schema files are found.
- **Unified events timeline.** Native task events and custom-table rows are merged
  into a single onset-ordered `events.tsv` at the session root, with an
  `events.json` sidecar generated from the custom-table schema.
- **`sourcedata/` provenance tier.** Each raw recording folder is copied verbatim,
  organized by subject and session.
- **`clock_dropout` validation check**, registered in the default check set.
- **Engine-agnostic software/OS version metadata.** Any scalar `*version*` key in
  session metadata is captured and surfaced in the HTML report header and the BIDS
  `SoftwareVersions` sidecar, with curated labels for known vendor keys (Unity,
  OVR Plugin, Horizon OS).
- **Custom-table schema/CSV consistency logging** in `load_session`.

### Changed

- Custom tables load from a per-source-dir folder (`custom_tables_dir`) instead of
  glob patterns.
- Events use a `name` column end-to-end, including the HTML report timeline.
- BIDS configuration now ships default `authors` and `readme_text`.
- The package version is single-sourced via `importlib.metadata`.
- The bundled demo dataset was replaced with Binary Choice, Maze Navigation, and
  Museum Viewing sessions (2026.06.10); BIDS output and HTML reports were
  regenerated.

### Fixed

- Quality-flag boundaries now use the global `timeSinceStartup` clock.
- Per-session `CustomTables` folder handling and events column typing.
- Validation of shared custom-column metadata across tables, and input
  file-format handling.
- Robust BIDS `README` and authors metadata generation.
- `eyes_closed` detection now runs on sessions with face tracking but no eye
  tracking (it previously required an EYES stream and was skipped entirely).
- The HTML report timeline now reflects the merged events timeline (native +
  custom-table events), matching `events.tsv`.
- The events.json `name` sidecar no longer lists the BIDS `n/a` token (or NaN)
  as a categorical level.
- A missing optional `CustomTables` folder is logged at debug level instead of
  error.
- All BIDS outputs (motion/channels/events TSV, JSON sidecars, README,
  `.bidsignore`, and HTML reports) are written with LF line endings on every
  operating system, so generated datasets are byte-reproducible across Windows
  and Linux. A repository `.gitattributes` rule normalizes text files to LF.

### Documentation

- Corrected the Conda activation command in the README user install
  (`conda activate resxr` → `conda activate resxr_env`).
- Fixed the README config reference: `bids.missing_values` example (`NaN` →
  `n/a`); validation thresholds and `check_column_groups` shown nested under
  `settings:`; input glob patterns; added `clock_dropout`, `readme_text`, and
  `task_description`; clarified that `load_session()` alone does not populate
  `total_duration_seconds`.
- Corrected the repository URLs (`ResXR/ResXR` → `ResXR/resxr-python-pipeline`)
  in `pyproject.toml` and the README clone/citation snippets.

## [0.1.1] - 2026-04-29

### Fixed

- Trailing-zero timestamps in the sampling-rate check; the validation window is
  computed from recording onset/offset.
- BIDS events export filename and column order.
- `pyproject` metadata layout, plus CI packaging and Ruff cleanup.

### Documentation

- Clarified optional events CSV export.

## [0.1.0] - 2026-04-29

First public release of the ResXR Python pipeline:

- IO and per-system stream splitting.
- Quality validation (hand-tracking-loss, sampling-rate, eye-closure, stats).
- Preprocessing with quality-flag masking (NaN replacement, no row deletion).
- Motion-BIDS export (motion/channels/events TSV + JSON sidecars).
- Self-contained HTML quality report.
- Command-line interface and programmatic Python API.

[Unreleased]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ResXR/resxr-python-pipeline/releases/tag/v0.1.0
