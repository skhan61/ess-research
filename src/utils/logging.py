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
import sys
from pathlib import Path


# ── Format used by setup_logging (not pytest — pytest has its own format) ────
_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

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
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_FMT,
        datefmt=_DATE_FMT,
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
