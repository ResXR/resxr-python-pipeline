# Changelog

All notable changes to the ResXR Python pipeline are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.2.2] - 2026-10-04

### Fixed

- The shipped configuration loads: an empty `validation.settings` block is read as `{}`.
- Event onsets in `events.tsv` are relative to the session's motion time zero (the first valid `timeSinceStartup`, where `latency_global` is 0).
- Quality masking (only with `apply_quality_masking: true`): hand tracking loss masks that hand's measurements and keeps its status columns; eye closure masks only the Eyes gaze values, not the Face stream or the head-tracked `Node_EyeCenter` pose.
- The HTML quality report embeds plotly.js once and loads nothing from the internet.
- In a stream with a selected hardware clock, the other `*_Time` columns are converted to that clock's time base.
- The package version is 0.2.2 in `pyproject.toml` and `uv.lock`.
- `acq_time` in `scans.tsv` keeps the recorded fraction of a second and the `Z` that marks UTC, and is read on Python 3.10.
- Hand-joint channels (`Left_XRHand_*`, `Right_XRHand_*`) in `channels.tsv` name the frame the session recorded them in, detected per hand: `global` when the Wrist joint equals the hand root, `tracking_space` when the joints are the SDK's raw tracking-space values (as in the demo data), or `n/a` with a warning; `channels.json` describes `tracking_space` when it is used, and the quality report names the frame. Values are unchanged.
- A session in which a stream cannot be timed (its clock or its `timeSinceStartup` column holds a non-numeric value, its clock has no nonzero value or fewer than two distinct times, the stream has a single row, or the face file has neither `timeSinceStartup` nor `Face_Time`) is logged as an error and skipped, and the run continues with the other sessions; before, it stopped the whole run.
- `latency` and `latency_global` are `n/a` in every row whose clock value is zero or not finite, also between valid values (before, only rows before the first and after the last valid value were `n/a`, and a zero inside the recording gave a negative latency).
- With `output.overwrite: true`, a rerun replaces a `sourcedata/` copy whose files are read-only (before, every such session was skipped with "Access is denied").
- With `report.output_dir` set, two sessions with the same `session_id` write one report file (the second replaces the first); a warning names both sessions.
- A session read from its own copy in the output's `sourcedata/` (reprocessing from the copies) is not copied onto itself: an info line says that sourcedata is the input folder and nothing was copied (before, with `output.overwrite: true`, every such session was skipped with "are the same file").
- A face file that cannot be split into columns, or that has neither `timeSinceStartup` nor the configured Face clock column (as a semicolon-separated file), gets the same format hint as the continuous file: comma-separated, `.` as the decimal separator, and a header row that includes `timeSinceStartup` (before, the messages named the missing `Face_Time` and `timestamp` columns). As before, the first is read as a missing face file and the second skips its session.
- A metadata `session_id` with a character other than letters (A-Z, a-z), digits, `.`, `_` and `-` skips its session with an error that names the key and these characters (`session_id` names the report and the custom-tables folder; before, `"../../../evil"` wrote the report outside the BIDS root, and on Windows a drive letter such as `"M:evil"` sent it to that drive).
- `participants.tsv` has one row per subject (before, one row per processed session: a subject with two sessions, such as `ses-01` and `ses-02`, was listed twice).

### Changed

- The pipeline reports every departure from the input contract at the default log level: it converts and warns when the intent is clear, continues without the affected stream or check and warns when expected data are missing, and skips the session with an error when the data cannot be trusted. An unexpected error in one session also skips only that session (the traceback is shown with `-v`), and the run continues with the others. One session's problem never stops the run. A skipped session leaves none of its files in the output: what the run had written for it (its `sourcedata/` copy, raw and derivative session files, and report) is removed, together with the `sub-*`, `sourcedata/`, `derivatives/resxr/` and `derivatives/` folders it created and left empty, and an info line names what was removed; files marked read-only are removed too. Each session folder or report of a skipped session that existed before the session started is kept, with one warning per folder: that it may now hold partial files, or, when the session did not write into it, that it holds output of an earlier run (none for a session's input folder, when the session is read from its copy in `sourcedata/`). A folder that cannot be checked or removed is a warning naming it, and the run continues. The same cleanup applies to a session interrupted with Ctrl-C (the run still stops, exit code 130) and to a session that fails in a direct call of `process_session_from_mapping` (before, a session skipped after its first write kept those files, though it was not in `participants.tsv`).
- Exit code: `resxr` exits 1 when a session was skipped, when no session was found, or when the configuration is invalid, and 0 only when every session found was processed. `resxr.run()` returns a `RunResult` listing each session as processed (with its warning count and the checks that did not run) or skipped (with the reason), and every run ends with a summary of the same.
- These configuration errors stop the run before any session is read, and `--dry-run` reports them: a key the configuration does not define, at any level, named with its location (before, an unknown top-level key was a warning and an unknown key below the top level was ignored without a message), a setting name under `validation.settings` that no registered check reads (before, it was ignored and the checks kept their defaults; each check now lists the settings keys it reads in a `settings_keys` attribute, and a custom check that reads a settings key must list it there; keys inside a setting's value, such as the hand names under `tracking_flags`, are not checked), a column-group name under `validation.settings.check_column_groups` that `validation.column_groups` does not define (before, accepted without a message), unknown check names in `validation.enabled_checks` or `preprocessing.masking_checks`, unknown tracking-system names under `systems`, `sampling_frequencies`, `system_descriptions` or `preprocessing.alternate_time_columns`, sampling rates that are not positive, a tracking system not disabled under `systems` that has no `sampling_frequencies` entry (before, `--dry-run` accepted it and the run skipped every session), with `apply_quality_masking: true`, `masking_checks` names that are not in `validation.enabled_checks` (they never run, so nothing would be masked for them), a `subject_id`, `session_label` or `output.task_name` that is not a BIDS label (letters and digits only; before, `subject_id: "sub-01"` wrote `sub-sub-01`, and a label containing `..` wrote files outside the BIDS root), two `session_mappings` with the same `subject_id` and `session_label` or the same folder (before, the two sessions' files were mixed in one session folder and `participants.tsv` listed the subject twice), and `session_mappings` of one `subject_id` that give different `age`, `sex` or `handedness` values, including a value given in one mapping and left out of another (`participants.tsv` has one row per subject). An `input.data_dir` that is not an existing folder is also an error at start-up (before, `--dry-run` accepted it, and the run skipped each mapped session or, with auto-discovery, stopped with "Pipeline failed"). `--dry-run` runs the same checks as a run and reports an error in the same form: `Configuration error in <config file>: ...`.
- The shipped configuration's comment on `output.overwrite` says what it protects (the behavior is unchanged): with `false`, a session's existing `sourcedata/` copy and the existing `README` and `dataset_description.json` files (raw and derivatives) are kept; session files (motion, channels, events, scans), `participants.tsv`/`.json` and reports are always rewritten, and `.bidsignore` is written only when absent (before, the comment read "Overwrite existing files").
- `*_enabled` values in `SessionMetadata.json` must be JSON `true` or `false`. The strings `"true"`/`"false"` are converted with a warning; any other value counts as false with a warning (before, any non-empty string, including `"false"`, enabled the stream).
- A stream that has data but is not exported says why: the metadata key is missing or not a boolean (warning), the metadata or the config disables it (info), its columns are absent, or the face file is missing although `face_enabled` is true (warning). A configured per-system clock column that is absent, a missing, `null` or empty `session_id` or one that is not text (read as `unknown`; before, `false` and `123` became the ids `False` and `123`), a `fixedDeltaTime` that is not a number (0.02 s is used; before, a string such as `"0.02"` stopped the report step), and an unparseable `utc_start_iso8601` are warnings that name the consequence.
- Continuous-data columns that no stream takes are listed (count and up to ten names): columns with no tracking-system prefix and Face columns in the continuous file (warning), and `Custom_` transform columns (info). They stay in `sourcedata/`.
- `timeSinceStartup` is checked when read, without changing it: warnings when its median step exceeds 0.5 s (looks like milliseconds), when it steps backward, when cells are empty, NaN or infinite, when values between valid values are zero, and when only one row has a finite nonzero value (a column with none skips the session with an error); an info line per file counts the gaps (steps longer than 10 times the expected Head interval, one over `sampling_frequencies.Head`, or 10 times the median step when no Head rate is configured) and names the longest, except when the column looks like milliseconds. Rows are not interpolated. A stream's own clock column (`preprocessing.alternate_time_columns`, also `Face_Time` in the face file) is checked for milliseconds (median step between distinct values above 0.5 s) and backward steps, with warnings naming the column (before, it was not checked); its empty or zero cells between valid values are flagged by the `clock_dropout` check, as before.
- The sampling-rate check's irregularity test (coefficient of variation of the intervals above `sampling_cv_threshold`, default 0.5) measures the intervals between the stream's own samples: the distinct finite nonzero values of its clock column from the recording onset, as the effective rate does. Before, it measured the intervals between rows on `timeSinceStartup`, where the template writes rows in per-frame bursts, so it flagged every stream of every session (demo: 12 such flags before, 4 after, all in the Maze session). The flag still spans the recording on `timeSinceStartup`.
- A missing events file is reported (info, or a warning naming a similarly named CSV that was not read); a custom-tables folder whose name does not match `{session_id}_CustomTables`, and `events.tsv` columns that `events.json` does not describe, are warnings. Auto-discovery that finds no session is an error that names deeper folders holding a metadata file. A continuous file with a header and no rows skips its session with an error.
- Quality checks that cannot run are logged and listed in the HTML report under "Checks not run" (check, stream, reason): `hands_tracking_loss` without its flag column, or with flag values other than 0/1, true/false or blank (a warning, and no hand-loss flags are made from such a column; before, only blank rows counted as lost), `eyes_closed` without `Eyes_Closed_L`/`Eyes_Closed_R`, a check whose stream is absent or has no separate `timeSinceStartup` column, and a check that failed. A stream counts as absent only when it was expected: a stream turned off on purpose (`systems.<name>.enabled: false` in the config, or JSON `false` in its `*_enabled` metadata key) is not listed, and leaves the run summary at info. The reason names only the expected streams and says why each is absent, e.g. "the session has no Hands stream (metadata: hands_enabled missing; no Hands columns in the continuous data)".
- The shipped configuration's comments say that `sampling_frequencies` holds the expected rate per tracking system in Hz (an integer is accepted), that each `validation.settings` key must be declared by a registered check, and what `alternate_time_columns: {}` does: every stream is timed by `timeSinceStartup` alone, and the sampling-rate, clock-dropout, hand-tracking-loss and eye-closure checks log an error for each stream they check and raise no flags. The CLI help names the repository `ResXR/resxr-python-pipeline` (before, `ResXR/ResXR`) and says that `-q` hides warnings and the run summary.

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

[Unreleased]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.2.1...v0.2.2
[0.2.0]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ResXR/resxr-python-pipeline/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ResXR/resxr-python-pipeline/releases/tag/v0.1.0
