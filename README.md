# ResXR Python Pipeline

Converts behavioral and sensor recordings from Unity/Meta Quest XR experiments into [Motion-BIDS](https://bids.neuroimaging.io/) datasets. Each session folder of tracking CSVs and a `SessionMetadata.json` becomes a standardized dataset with per-system motion files, channel descriptions, JSON sidecars, merged events, and an HTML quality report. It is a Python package (≥3.10) with a `resxr` command-line tool and a small programmatic API.

This pipeline is the data-processing half of ResXR; the recordings it reads are produced by the [`resxr-unity-research-template`](https://github.com/ResXR/resxr-unity-research-template). The two share no code — only the [input file format](https://docs.resxr.org/python/input-data/).

## 📖 Full documentation

**https://docs.resxr.org** (Python Pipeline section) — installation, configuration, input/output formats, validation, the API, and more. The pages below are the source of truth; this README is only a quick start.

## Install

Clone the repository, then set up an environment with conda **or** uv.

**Conda:**

```bash
git clone https://github.com/ResXR/resxr-python-pipeline.git
cd resxr-python-pipeline
conda env create -f environment.yml      # or environment-dev.yml for dev tools
conda activate resxr_env                 # resxr_dev for the dev environment
```

**uv:**

```bash
git clone https://github.com/ResXR/resxr-python-pipeline.git
cd resxr-python-pipeline
uv sync --all-extras
```

See [Installation](https://docs.resxr.org/python/installation/) for details and requirements.

## Run

The repository ships demo data under `DATA/` and a ready-to-run config, so you can process it immediately:

```bash
resxr -c config/pipeline_config.yaml            # run the pipeline
resxr -c config/pipeline_config.yaml --dry-run  # validate config without writing
```

Equivalent invocations: `python -m resxr -c config/pipeline_config.yaml`, or from Python:

```python
import resxr
resxr.run(config_path="config/pipeline_config.yaml")
```

Follow the [Quickstart](https://docs.resxr.org/python/quickstart/) for a full walkthrough.

## Documentation map

| Topic | Page |
| ----- | ---- |
| Input file format (the data contract) | [Input Data Format](https://docs.resxr.org/python/input-data/) |
| Every config option | [Configuration](https://docs.resxr.org/python/configuration/) |
| The six processing stages | [Pipeline Stages](https://docs.resxr.org/python/pipeline-stages/) |
| Quality checks & custom checks | [Validation](https://docs.resxr.org/python/validation/) |
| Output dataset layout | [BIDS Output](https://docs.resxr.org/python/bids-output/) |
| HTML quality reports | [Quality Reports](https://docs.resxr.org/python/reports/) |
| Command-line options | [CLI Reference](https://docs.resxr.org/python/cli/) |
| Programmatic API | [Python API](https://docs.resxr.org/python/api/) |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, and open a pull request. See [CHANGELOG.md](CHANGELOG.md) for release history.

## Citation

```bibtex
@software{resxr,
  title = {ResXR: XR Experiment Data Processing Pipeline},
  year  = {2026},
  url   = {https://github.com/ResXR/resxr-python-pipeline}
}
```
