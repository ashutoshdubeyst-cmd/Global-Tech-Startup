"""Central logging configuration for the churn-model project."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = "churn_model.log"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MANAGED_HANDLER_ATTRIBUTE = "_churn_model_handler"


def _convert_log_level(level: int | str) -> int:
    """Convert a numeric or text log level into a logging constant."""

    if isinstance(level, int):
        return level

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level '{level}'. "
            "Use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )
    return numeric_level


def setup_logging(
    logger_name: str | None = None,
    *,
    level: int | str = logging.INFO,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    log_filename: str = DEFAULT_LOG_FILE,
    console: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure project logging and return the requested logger.

    Messages are written to a rotating file and, by default, the terminal.
    Calling this function more than once does not create duplicate handlers.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    if backup_count < 0:
        raise ValueError("backup_count cannot be negative")
    if not log_filename.strip():
        raise ValueError("log_filename cannot be empty")

    numeric_level = _convert_log_level(level)
    log_directory = Path(log_dir).expanduser().resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / log_filename

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove only handlers previously created by this function. This keeps the
    # configuration idempotent without disturbing handlers owned by libraries.
    for handler in root_logger.handlers[:]:
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
    root_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    return logging.getLogger(logger_name)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger after logging has been configured."""

    return logging.getLogger(name)


if __name__ == "__main__":
    logger = setup_logging(__name__)
    logger.info("Logging configured successfully.")

