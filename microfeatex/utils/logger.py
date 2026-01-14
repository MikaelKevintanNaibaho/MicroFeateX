"""Centralized logging configuration for MicroFeatEX.

This module provides a consistent logging interface across the entire project.
Log level can be configured via the MICROFEATEX_LOG_LEVEL environment variable.

Example:
    from microfeatex.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import os
import sys
from typing import Optional

__all__ = ["get_logger", "setup_file_logging"]

# Default log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cache for loggers to avoid duplicate handlers
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Optional log level override. If not provided, uses
               MICROFEATEX_LOG_LEVEL env var or defaults to INFO.

    Returns:
        Configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Determine log level
    if level is None:
        env_level = os.environ.get("MICROFEATEX_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    logger.setLevel(level)

    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate logs)
    logger.propagate = False

    _loggers[name] = logger
    return logger


def setup_file_logging(
    logger: logging.Logger, log_file: str, level: int = logging.DEBUG
) -> None:
    """Add file handler to an existing logger.

    Args:
        logger: Logger instance to add file handler to.
        log_file: Path to the log file.
        level: Log level for file output (default: DEBUG).
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
