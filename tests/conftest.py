"""Shared pytest configuration and test-data fixtures."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import setup_logging


@pytest.fixture
def labelled_startups_path() -> Path:
    """Return the labelled cleaned dataset used for training tests."""

    return TEST_DATA_DIR / "startups_cleaned.csv"


@pytest.fixture
def new_startups_path() -> Path:
    """Return the unlabelled cleaned dataset used for inference tests."""

    return TEST_DATA_DIR / "new_startups_cleaned.csv"


@pytest.fixture
def configured_test_logging(tmp_path: Path):
    """Configure file logging and close its handlers after the test."""

    root_logger = logging.getLogger()
    original_level = root_logger.level
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir, console=False)

    try:
        yield log_dir / "churn_model.log"
    finally:
        for handler in root_logger.handlers[:]:
            if getattr(handler, "_churn_model_handler", False):
                root_logger.removeHandler(handler)
                handler.close()
        root_logger.setLevel(original_level)
