"""
utils/logger.py
===============
Structured logging: simultaneous output to console (human-readable) and to a
rotating JSON-lines file (machine-parseable).

Usage
-----
    from utils.logger import get_logger

    log = get_logger("server", log_dir="results/logs/run_001/")
    log.info("Round %d complete", round_idx)
"""

import json
import logging
import logging.handlers
import os
import time
from typing import Optional


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _ConsoleFormatter(logging.Formatter):
    """Coloured, human-readable console format."""

    _COLOURS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self._COLOURS.get(record.levelname, "")
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        prefix = f"{colour}[{ts}] [{record.levelname:8s}] [{record.name}]{self._RESET}"
        return f"{prefix}  {record.getMessage()}"


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return (or create) a named logger with console + optional file handlers.

    Parameters
    ----------
    name    : str
        Logger name (e.g. ``"server"``, ``"client.0"``).
    log_dir : str, optional
        Directory where ``<name>.jsonl`` will be written.  If ``None``, file
        logging is disabled.
    level   : int
        Logging level (default ``INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(_ConsoleFormatter())
    logger.addHandler(ch)

    # File handler (JSON lines, rotating at 5 MB, keep 3 backups)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name.replace('.', '_')}.jsonl")
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)  # capture everything in file
        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)

    return logger


def configure_root_logger(level: int = logging.WARNING) -> None:
    """
    Set the root logger level to suppress noisy third-party library logs.
    Call once at application start (``main.py``).
    """
    logging.basicConfig(level=level)
    # Suppress overly verbose libraries
    for lib in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)
