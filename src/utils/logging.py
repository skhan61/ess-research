"""
Central logging configuration for the ESS-Research project.

Usage in any module::

    from src.utils.logging import get_logger
    logger = get_logger(__name__)

    logger.info("training started")
    logger.debug("batch shape: %s", batch["image"].shape)

Call ``setup_logging()`` once at the start of a training/inference script.
Tests do NOT call ``setup_logging()`` — pytest handles log output via
``log_cli`` settings in ``pyproject.toml``.
"""

import logging
import re
import sys
from pathlib import Path

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ── Format used by setup_logging (not pytest — pytest has its own format) ────
_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"

_LEVEL_COLORS = {
    "DEBUG":    "\033[34m",   # blue
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}


class _ColorFormatter(logging.Formatter):
    """Adds ANSI color to levelname when writing to a TTY."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{_BOLD}{record.levelname:<8}{_RESET}"
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter that strips any ANSI escape codes from the message."""

    def format(self, record: logging.LogRecord) -> str:
        record.msg = _ANSI_RE.sub("", str(record.msg))
        record.levelname = _ANSI_RE.sub("", record.levelname)
        return super().format(record)

# Noisy third-party loggers we always silence
_QUIET_LOGGERS = (
    "PIL",
    "torch",
    "torchvision",
    "lightning",
    "lightning_utilities",   # silences 💡 Tip: upgrade messages
    "transformers",
    "huggingface_hub",
    "filelock",
    "urllib3",
)


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """
    Configure the root logger for the whole project.

    Call this once at the top of a training or inference script::

        from src.utils.logging import setup_logging
        setup_logging(level=logging.INFO, log_file=Path("logs/run.log"))

    Args:
        level:    Logging level for project code (default INFO).
        log_file: Optional path to write logs to disk in addition to stdout.
                  Parent directories are created automatically.
    """
    stream = logging.StreamHandler(sys.stdout)
    # Use color formatter only when stdout is a real terminal (not redirected)
    if sys.stdout.isatty():
        stream.setFormatter(_ColorFormatter(fmt=_FMT, datefmt=_DATE_FMT))
    else:
        stream.setFormatter(logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT))

    handlers: list[logging.Handler] = [stream]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Log file always plain text (no ANSI)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(_PlainFormatter(fmt=_FMT, datefmt=_DATE_FMT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # override any previously installed root handlers
    )

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Use at module level::

        logger = get_logger(__name__)

    Works in both test (pytest log_cli) and production (setup_logging)
    contexts because Python logging always propagates to the root logger.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
