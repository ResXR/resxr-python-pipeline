"""
Command-line interface for the ResXR pipeline.

Provides the main entry point for running the XR data processing pipeline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the ResXR CLI."""

    parser = argparse.ArgumentParser(
        prog="resxr",
        description="ResXR - VR experiment data processing pipeline. "
        "Converts XR tracking data to BIDS format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  resxr -c config/pipeline_config.yaml
  resxr --config my_config.yaml --dry-run
  resxr --version

For more information, see: https://github.com/ResXR/ResXR
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to the pipeline YAML configuration file. (default: config/pipeline_config.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print what would be processed without writing output.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging.",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except errors.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ResXR {__import__('resxr').__version__}",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for the `resxr` command-line interface.

    This function is called from:
      * `python -m resxr` (via __main__.py)
      * a console_script entry point in pyproject.toml

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors)
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on verbosity
    import logging

    from .core.logger import setup_logging

    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    setup_logging(level=log_level)
    logger = logging.getLogger("resxr")

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - validating configuration...")
        try:
            from .core.config import PipelineConfig

            config = PipelineConfig.from_yaml(config_path)
            logger.info(f"Configuration valid: {config_path}")
            logger.info(f"  Input directory: {config.input.data_dir}")
            logger.info(f"  Output directory: {config.output.bids_root}")
            logger.info(f"  Session mappings: {len(config.session_mappings)}")
            logger.info(f"  Enabled systems: {[k for k, v in config.systems.items() if v.enabled]}")
            return 0
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return 1

    # Run pipeline
    try:
        pipeline.run(config_path=str(config_path))
        return 0
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
