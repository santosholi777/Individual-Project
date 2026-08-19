"""Logging setup for the DeepVisionAttend AI service.

Provides a single :func:`setup_logging` entry point used by both the FastAPI
application and the command line tools, so log output is identical regardless of
how the service is driven.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-28s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Third-party loggers that are far too chatty at DEBUG level.
_NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3", "onnxruntime", "numba")

_configured = False


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    *,
    log_to_file: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging for the process.

    Safe to call more than once: subsequent calls are ignored unless ``force``
    is set, which prevents duplicated handlers (and duplicated log lines) when
    both a CLI entry point and an imported module configure logging.

    Args:
        level: Root log level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        log_dir: Directory for the rotating log file. Required when
            ``log_to_file`` is true.
        log_to_file: Whether to additionally write a rotating file log.
        force: Reconfigure even if logging was already set up.

    Returns:
        The configured root logger.
    """
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    if log_to_file and log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / "ai-service.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
            )
            root.addHandler(file_handler)
        except OSError as exc:
            # A read-only or missing volume must not stop the service booting;
            # console logging is enough to diagnose it.
            root.warning("File logging disabled (%s): %s", log_dir, exc)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Modules should call ``get_logger(__name__)`` at import time; handler
    configuration is the entry point's responsibility, not the module's.
    """
    return logging.getLogger(name)
