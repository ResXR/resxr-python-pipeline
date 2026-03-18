"""
Logging configuration for the ResXR pipeline.

Provides structured logging with configurable verbosity levels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO, log_file: Path | None = None, name: str = "resxr"
) -> logging.Logger:
    """
    Configure logging for the ResXR pipeline.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO)
    log_file : Optional[Path]
        If provided, also log to this file
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "resxr") -> logging.Logger:
    """
    Get an existing logger by name.

    Parameters
    ----------
    name : str
        Logger name (use module name for hierarchical logging)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
