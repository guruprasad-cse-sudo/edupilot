"""
EduPilot AI Faculty Assistant
Module: logging_utils.py
Version: 4.0.0
Author: EduPilot Team
Purpose: Reusable structured logging setup. Provides a single get_logger()
         factory used by every module to guarantee consistent format, level,
         and handler configuration across the application.
"""

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level flag to avoid duplicate handler registration
# ---------------------------------------------------------------------------
_ROOT_CONFIGURED: bool = False


def _configure_root_logger(level: int) -> None:
    """Configure the root logger once with a StreamHandler to stdout.

    Idempotent — subsequent calls are no-ops once configured.

    Args:
        level: Logging level integer (e.g. logging.INFO).
    """
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=date_fmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Prevent duplicate handlers if a framework (e.g. Streamlit) already
    # attached one.
    if not root.handlers:
        root.addHandler(handler)

    _ROOT_CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a named logger, initialising the root logger on first call.

    The logging level is resolved in order:
      1. Explicit *level* argument passed to this function.
      2. ``LOG_LEVEL`` from the central config (imports lazily to avoid
         circular import issues at startup).
      3. ``INFO`` as the hard fallback.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Optional override level string (e.g. ``"DEBUG"``).

    Returns:
        A configured :class:`logging.Logger` instance.

    Example::

        from logging_utils import get_logger
        logger = get_logger(__name__)
        logger.info("EduPilot started")
    """
    # Lazy import to break potential circular dependency at startup
    resolved_level_str: str = "INFO"
    try:
        from config import config as _cfg  # noqa: PLC0415
        resolved_level_str = _cfg.log_level
    except Exception:  # pragma: no cover
        pass

    if level is not None:
        resolved_level_str = level.upper()

    numeric_level = getattr(logging, resolved_level_str, logging.INFO)
    _configure_root_logger(numeric_level)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    return logger
