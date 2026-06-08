# ResXR

**VR Experiment Data Processing Pipeline**

An open-source toolkit for standardized XR behavioral research. Converts Unity/Meta Quest tracking data to [BIDS](https://bids.neuroimaging.io/) (Brain Imaging Data Structure) compliant format.

---

## Features

- 🎯 **Multi-system tracking support**: Head, Hands, Eyes, Face, Body, Controllers
- 📊 **BIDS-compliant output**: Generates motion.tsv, channels.tsv, events.tsv, and JSON sidecars; **LATENCY channels** (per-system and global `timeSinceStartup`) for per-sample timing from recording onset
- ✅ **Quality validation**: Tracking loss, sampling irregularities; **column-specific flagging** (e.g. flag only left-hand columns when left hand loses tracking)
- 📈 **HTML reports**: Visual quality reports; **quality flag times in global time** (converted to `timeSinceStartup` and relative to recording onset)
- ⚙️ **Configurable pipeline**: YAML-based configuration; **alternate time columns** per system (e.g. Hands → `Node_HandLeft_Time`); optional **quality masking** (NaN replacement, no row deletion)
- 🔧 **Multi-stream validation**: Checks can declare `required_streams` and access other streams via `session.get_stream()`
- 🔧 **CLI & programmatic API**: Use from command line or import as a library

---

## Installation

### Option 1: Using Conda (Recommended)

**For users (minimal install):**

```bash
git clone https://github.com/ResXR/ResXR.git
cd ResXR
conda env create -f environment.yml
conda activate resxr
```

**For developers (includes dev tools):**

```bash
git clone https://github.com/ResXR/ResXR.git
cd ResXR
conda env create -f environment-dev.yml
conda activate resxr_dev
```

The dev environment includes pytest, ruff, and installs the package in editable mode.

### Option 2: Using uv

```bash
git clone https://github.com/ResXR/ResXR.git
cd ResXR
uv sync --all-extras
```

Run tests:

```bash
uv run pytest
```

### Requirements

- Python ≥3.10 (3.12+ recommended)
- pandas, numpy, pyyaml, pydantic, plotly, jinja2

---

## Quick Start

### 1. Configure the pipeline

Edit `config/pipeline_config.yaml`:

```yaml
# Input/Output paths
input:
  data_dir: /path/to/your/session/data
  continuous_data_pattern: "*_ContinuousData.csv"
  face_data_pattern: "*_FaceExpressionData.csv"
  metadata_pattern: "*SessionMetadata.json"
  events_data_pattern: "*_Events.csv"  # Optional: task/stimulus events

output:
  bids_root: /path/to/output
  dataset_name: "My VR Study"
  task_name: "VRtracking"

# Map source folders to BIDS subject/session IDs
session_mappings:
  - source_dir: "session_001"
    subject_id: "01"
    session_label: "01"

# Hardware metadata
device:
  manufacturer: "Meta"
  model_name: "Meta Quest"
```

### 2. Run the pipeline

**CLI:**

```bash
resxr -c config/pipeline_config.yaml
```

**Python:**

```python
import resxr
resxr.run(config_path="config/pipeline_config.yaml")
```

### 3. Validate before processing (dry run)

```bash
resxr -c config/pipeline_config.yaml --dry-run
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ResXR Pipeline                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Data      │───▶│   Split by       │───▶│   Validate      │
│   (CSV + JSON)   │    │   Tracking       │    │   Quality       │
│                  │    │   System         │    │                 │
└──────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Generate       │◀───│   Export BIDS    │◀───│   Preprocess    │
│   Reports        │    │   (prepare_      │    │   (optional     │
│   (global time)  │    │   motion_data +  │    │   NaN masking)  │
│                  │    │   write)         │    │                 │
└──────────────────┘    └──────────────────┘    └─────────────────┘
```

Before writing, each stream is passed through **prepare_motion_data**: internal time columns (`timestamp`, `timeSinceStartup`) are converted into BIDS **LATENCY** channels (`latency`, `latency_global` — seconds from recording onset) and the originals are removed. The HTML report shows quality flag times in **global time** (`timeSinceStartup`), relative to recording onset.

---

## Input Data Format

ResXR expects data recorded from Unity/Meta Quest with the following structure:

```
session_folder/
├── *_ContinuousData.csv      # Main tracking data (head, hands, eyes, etc.)
├── *_FaceExpressionData.csv  # Face tracking (FACS blendshapes) - optional
├── *_Events.csv              # Task/stimulus event markers - optional
├── *_SessionMetadata.json    # Recording configuration & timestamps
└── *_CustomTables/           # Experiment-defined custom data classes - optional
    ├── *_CustomTables.json   # Schema describing each custom table's columns
    └── *_<ClassName>.csv     # One CSV per custom data class
```

### Expected CSV columns

The pipeline automatically identifies tracking systems by column prefixes:


| System      | Column Prefixes                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Head        | `Node_Head_*`, `FocusedObject`, `RecenterCount`, `TrackingLost`, `UserPresent`, `timeSinceStartup`, `TrackingOriginChange_*`, `TrackingTransform_*` |
| Hands       | `Node_HandLeft_*`, `Node_HandRight_*`, `LeftHand_*`, `RightHand_*`, `Left_XRHand_*`, `Right_XRHand_*`                                               |
| Eyes        | `EyeGazeHitPosition_*`, `RightEye_*`, `LeftEye_*`, `Node_EyeCenter_*`, `Eyes_Time`                                                                  |
| Face        | `Face_*`, `Brow_*`, `Cheek_*`, `Jaw_*`, `Lip_*`, `Lips_*`, `Tongue_*`, etc.                                                                         |
| Body        | `Body_*` (e.g. `Body_Time`, `Body_Confidence`, `Body_Chest_px`, `Body_Head_Flags_*`)                                                                |
| Controllers | `Node_ControllerLeft_*`, `Node_ControllerRight_*`                                                                                                   |


### Events CSV Format (Optional)

If using task/stimulus events, provide a CSV with these columns:


| Column     | Type   | Description                                                  | Example                                           |
| ---------- | ------ | ------------------------------------------------------------ | ------------------------------------------------- |
| `onset`    | float  | Event start time in seconds (relative to session start)      | `2.5`, `5.832`                                    |
| `duration` | float  | Event duration in seconds (use `0` for instantaneous events) | `1.5`, `0`                                        |
| `name`     | string | Event type/label; exported as BIDS `trial_type`              | `"stimulus_onset"`, `"response"`, `"trial_start"` |


**Example events CSV:**

```csv
onset,duration,name
0.0,0,trial_start
2.5,1.5,stimulus_A
4.2,0,response
5.0,0,trial_end
```

**Note:** Duration of `0` indicates an instantaneous event (e.g., button press, stimulus onset).

At BIDS export, ResXR writes events to the session `motion/` directory using the configured task name, for example `sub-01_ses-01_task-VRtracking_events.tsv`. The input `name` column is exported as BIDS `trial_type`, and the output `events.tsv` columns start with BIDS-required `onset`, `duration`, then `trial_type`.

---

## BIDS Output Structure

```
bids_root/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── .bidsignore
└── sub-01/
    └── ses-01/
        ├── sub-01_ses-01_scans.tsv
        └── motion/
            ├── sub-01_ses-01_task-VRtracking_tracksys-Head_motion.tsv
            ├── sub-01_ses-01_task-VRtracking_tracksys-Head_motion.json
            ├── sub-01_ses-01_task-VRtracking_tracksys-Head_channels.tsv
            ├── sub-01_ses-01_task-VRtracking_tracksys-Head_channels.json
            ├── sub-01_ses-01_task-VRtracking_tracksys-Hands_motion.tsv
            ├── sub-01_ses-01_task-VRtracking_events.tsv  # Optional: task events
            ├── sub-01_ses-01_task-VRtracking_events.json
            └── ... (similar for Eyes, Face, etc.)
```

---

## Configuration Reference

### BIDS Specification Settings

All BIDS metadata values are configurable:

```yaml
bids:
  missing_values: "NaN"      # How NaN values are written
  dataset_type: "raw"        # BIDS dataset type
  license: "CC0"             # Dataset license
  authors: []                # List of authors
  reference_frame:           # Coordinate system
    description: "Global VR playspace coordinate system..."
    rotation_rule: "left-hand"
    rotation_order: "ZXY"
    spatial_axes: "RSA"
```

### Sampling Frequencies

Expected sampling rates for each tracking system (Hz):

```yaml
sampling_frequencies:
  Head: 72.0
  Hands: 90.0
  Eyes: 30.0
  Face: 30.0
  Body: 72.0
  Controllers: 90.0
```

### System Descriptions

Task descriptions for BIDS sidecar files:

```yaml
system_descriptions:
  Head: "Head position and rotation tracking from Meta Quest VR headset"
  Hands: "Hand and finger tracking from Meta Quest VR headset"
  Eyes: "Eye gaze tracking from Meta Quest VR headset"
  Face: "Facial expression tracking using FACS blend shapes"
  Body: "Full body joint tracking from Meta Quest VR headset"
  Controllers: "VR controller position and rotation tracking"
```

### Input Options


| Option                    | Description                            | Required |
| ------------------------- | -------------------------------------- | -------- |
| `data_dir`                | Root directory containing session data | Yes      |
| `continuous_data_pattern` | Glob pattern for main tracking CSV     | Yes      |
| `face_data_pattern`       | Glob pattern for face tracking CSV     | Yes      |
| `metadata_pattern`        | Glob pattern for session metadata JSON | Yes      |
| `events_data_pattern`     | Glob pattern for optional events CSV   | No       |


### Output Options


| Option         | Description                      | Required |
| -------------- | -------------------------------- | -------- |
| `bids_root`    | Output directory for BIDS data   | Yes      |
| `dataset_name` | Dataset name in description.json | Yes      |
| `bids_version` | BIDS specification version       | Yes      |
| `task_name`    | Task name in filenames           | Yes      |
| `overwrite`    | Overwrite existing files         | Yes      |


### Device Configuration

```yaml
device:
  manufacturer: "Meta"
  model_name: "Meta Quest"
```

### Tracking Systems

Enable/disable specific tracking systems:

```yaml
systems:
  Head: { enabled: true }
  Hands: { enabled: true }
  Eyes: { enabled: true }
  Face: { enabled: true }
  Body: { enabled: false }
  Controllers: { enabled: false }
```

### Validation Settings

```yaml
validation:
  sampling_rate_tolerance: 0.10       # Max deviation from expected rate (10%)
  sampling_cv_threshold: 0.5          # Max coefficient of variation (50%)
  eyes_closed_threshold: 0.9          # Blend shape value to consider eye closed (0-1)
  eyes_closed_min_duration: 0.1       # Min duration (seconds) to flag as closed
  enabled_checks:
    - hands_tracking_loss
    - sampling_rate
    - eyes_closed
    - stats_summary
  # Column groups for column-scoped checks
  column_groups:
    - name: "Left Hand"
      description: "Left hand wrist position and rotation"
      columns:
        - Left_XRHand_Wrist_x
        - Left_XRHand_Wrist_y
        - Left_XRHand_Wrist_z
    - name: "Right Hand"
      description: "Right hand wrist position and rotation"
      columns:
        - Right_XRHand_Wrist_x
        - Right_XRHand_Wrist_y
        - Right_XRHand_Wrist_z
  # Optional: assign which checks receive which groups (by group name)
  check_column_groups:
    hands_tracking_loss:
      - "Left Hand"
      - "Right Hand"
```

**Column groups** are defined by `name`, `description`, and `columns`. Each entry must have a `name` and `columns` key (validated at config load time). `check_column_groups` optionally assigns which checks receive which groups (by group name). If `column_groups` is omitted, checks fall back to a single default group containing all columns in the stream.

### Preprocessing Settings

Quality flags can optionally be applied as **NaN masks**. At BIDS export, **prepare_motion_data** converts internal time columns into BIDS **LATENCY** channels (seconds from recording onset) and strips the originals. — flagged values are replaced with NaN while all rows are preserved (no row deletion). This is disabled by default; the RAW dataset is never modified.

```yaml
preprocessing:
  apply_quality_masking: false        # Enable NaN masking for flagged segments
  masking_checks: null                # null = all flags; or list specific checks e.g. ["tracking_loss"]
  # Per-system time column. When set, splitter renames it to timestamp and keeps global as timeSinceStartup.
  # At write time: timestamp → latency (per-system), timeSinceStartup → latency_global. Leave {} for global only.
  alternate_time_columns:
    Hands: "Node_HandLeft_Time"
    # Eyes: "Eyes_Time"
    # Body: "Body_Time"
```

**Alternate time columns**: Some streams use their own time column (e.g. `Node_HandLeft_Time` for Hands). Map system name to exactly one column name; that column is used as the stream’s time axis and included in the stream’s columns.

### Report Configuration

```yaml
report:
  enabled: true
  output_dir: null   # null = auto (e.g. alongside BIDS output)
```

---

## Programmatic Usage

```python
from resxr import run, PipelineConfig, Session

# Option 1: Run with config file
run(config_path="config/pipeline_config.yaml")

# Option 2: Load and inspect config
from resxr.core.config import PipelineConfig
config = PipelineConfig.from_yaml("config/pipeline_config.yaml")
print(f"Input: {config.input.data_dir}")
print(f"Output: {config.output.bids_root}")

# Option 3: Access data structures directly (using config from YAML)
from resxr.io.readers import load_session
from resxr.core.config import PipelineConfig

config = PipelineConfig.from_yaml("config/pipeline_config.yaml")
session = load_session("/path/to/session_dir", config.input)
print(f"Session: {session.session_id}")
print(f"Duration: {session.total_duration_seconds:.1f}s")
```

---

## Quality Reports

When enabled, ResXR generates HTML reports including:

- **Session summary**: Duration, streams, quality flags (including column-specific flags with group labels when `column_groups` is used)
- **Timeline plot**: Interactive Plotly timeline with flagged segments and optional event markers
- **Per-stream stats**: Row counts, channel counts, NaN percentages, expected vs effective sampling rates
- **Quality flags table**: All flag times are in **global time** (`timeSinceStartup`), converted from per-stream time and expressed relative to recording onset (single shared timeline from 0)

---

## Project Structure

```
ResXR/
├── config/
│   └── pipeline_config.yaml    # Pipeline configuration (all values here!)
├── src/resxr/
│   ├── __init__.py             # Package exports
│   ├── cli.py                  # Command-line interface
│   ├── pipeline.py             # Main orchestration (calls prepare_motion_data at write)
│   ├── core/                   # Core data structures
│   │   ├── config.py           # PipelineConfig, ColumnGroup, ValidationConfig, etc.
│   │   ├── constants.py       # Enums & column patterns
│   │   ├── exceptions.py      # Custom exceptions
│   │   ├── logger.py          # Logging setup
│   │   └── session.py         # Session & TrackingStream
│   ├── io/                     # Input/Output
│   │   ├── readers.py          # CSV/JSON loaders
│   │   ├── splitter.py         # Split by tracking system (split-only; no LATENCY here)
│   │   ├── writers.py          # TSV/JSON writers (motion.tsv expects prepared data)
│   │   └── column_maps.py      # Column classification; LATENCY channel recognition
│   ├── bids/                   # BIDS formatting
│   │   ├── layout.py           # Directory structure
│   │   ├── metadata.py         # JSON sidecar (uses prepared_data when provided)
│   │   ├── naming.py           # BIDS filename conventions
│   │   └── channels.py         # channels.tsv from prepared DataFrame
│   ├── preprocessing/          # Data cleaning & BIDS prep
│   │   └── stream_preprocessing.py   # Masking + prepare_motion_data (LATENCY channels)
│   ├── validation/             # Quality checks
│   │   ├── registry.py         # Check registration
│   │   └── checks/             # Individual validators (extensible!)
│   │       ├── hands_tracking_loss.py  # Hand tracking loss detection
│   │       ├── sampling_rate.py        # Sampling irregularity detection
│   │       ├── eyes_closed.py          # Eye closure detection (face data)
│   │       └── stats.py               # Per-stream/column statistics
│   ├── utils/                  # Shared utilities
│   │   └── __init__.py         # find_recording_onset (onset = first non-zero time)
│   └── visualization/          # Reporting
│       ├── report.py           # HTML report (flag times in global timeSinceStartup)
│       └── templates/report.html
```

---

## Adding Custom Validation Checks

You can add custom validation checks in 3 steps:

1. Create a check class in `validation/checks/`:

```python
from resxr.core.config import ColumnGroup, ValidationConfig
from resxr.core.session import QualityFlag, Session, TrackingStream
from resxr.validation.registry import register_check

class MyCheck:
    name = "my_check"
    description = "Short description"
    required_streams = None   # None = per-stream; or [TrackingSystem.HANDS, ...] for multi-stream

    def __call__(self, stream, session, config):
        df = stream.data

        # Get column groups for this check (falls back to all columns if not configured)
        groups = config.get_column_groups(
            self.name,
            default_columns=[c for c in df.columns if c != "timestamp"],
        )

        flags = []
        for group in groups:
            # group.name        - human-readable label
            # group.columns     - list of column names
            # group.description - longer description
            ...
        return flags

# Register an instance of the check
register_check(MyCheck())
```

1. Export in `checks/__init__.py`
2. Enable in YAML:

```yaml
validation:
  enabled_checks:
    - my_check
```

**Column groups**: Checks call `config.get_column_groups(self.name)` to get their assigned `ColumnGroup` objects. If no groups are configured in the YAML, passing `default_columns` provides an automatic fallback (a single group with all stream columns). Groups are defined in `config.py` as a simple dataclass with `name`, `description`, and `columns` fields.

**Multi-stream checks**: Set `required_streams = [TrackingSystem.X, TrackingSystem.Y]`. The check runs once when processing the first stream in the list and can access other streams via `session.get_stream(system)`. Use this when a check needs data from more than one tracking system.

See the project [README](README.md) and source docstrings for details.

---

## License

Apache License 2.0 - see LICENSE file for details.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## Citation

If you use ResXR in your research, please cite:

```bibtex
@software{resxr,
  title = {ResXR: XR Experiment Data Processing Pipeline},
  year = {2026},
  url = {https://github.com/ResXR/ResXR}
}
```

