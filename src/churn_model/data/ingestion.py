"""Ingest source data into the project's raw-data directory.

This module copies the source file without changing it and then verifies that
the copied dataset can be read. Cleaning and feature engineering belong in
later pipeline stages.
"""

from __future__ import annotations

import argparse #reads command line arguments
import logging
import shutil #copies the original file to the raw-data directory
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionConfig:
    """Paths and options used by the ingestion step."""

    source_path: Path
    raw_data_dir: Path = Path("data/raw")
    overwrite: bool = False

    @property
    def destination_path(self) -> Path:
        """Return the location of the ingested raw file."""

        return self.raw_data_dir / self.source_path.name


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a supported tabular dataset into a DataFrame."""

    readers = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }

    suffix = path.suffix.lower()
    if suffix not in readers:
        supported = ", ".join(sorted(readers))
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: {supported}"
        )

    LOGGER.debug("Loading dataset from %s using the %s reader", path, suffix)
    return readers[suffix](path)


class DataIngestion:
    """Copy source data into ``data/raw`` and verify that it is readable."""

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config

    def run(self) -> tuple[Path, pd.DataFrame]:
        """Ingest the configured file and return its path and loaded data."""

        source = self.config.source_path.expanduser().resolve()
        destination = self.config.destination_path

        LOGGER.info(
            "Starting data ingestion: source=%s, destination=%s",
            source,
            destination,
        )

        if not source.is_file():
            raise FileNotFoundError(f"Source data file was not found: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Raw data already exists at {destination}. "
                "Set overwrite=True to replace it."
            )

        shutil.copy2(source, destination)
        data = load_dataset(destination)

        if data.empty:
            raise ValueError(f"The ingested dataset is empty: {destination}")

        LOGGER.info(
            "Ingested %s rows and %s columns into %s",
            len(data),
            len(data.columns),
            destination,
        )
        return destination, data


def ingest_data(
    source_path: str | Path,
    raw_data_dir: str | Path = "data/raw",
    *,
    overwrite: bool = False,
) -> tuple[Path, pd.DataFrame]:
    """Convenience function for running data ingestion."""

    config = IngestionConfig(
        source_path=Path(source_path),
        raw_data_dir=Path(raw_data_dir),
        overwrite=overwrite,
    )
    return DataIngestion(config).run()


def main() -> None:
    """Run ingestion from the command line."""

    parser = argparse.ArgumentParser(
        description="Copy a tabular dataset into the raw-data directory."
    )
    parser.add_argument("source", type=Path, help="Path to the source dataset")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing raw-data file",
    )
    args = parser.parse_args()

    destination, data = ingest_data(
        args.source,
        args.raw_data_dir,
        overwrite=args.overwrite,
    )
    print(f"Saved {len(data)} rows to {destination}")


if __name__ == "__main__":
    try:
        from src.logger import setup_logging
    except ModuleNotFoundError:
        # Support VS Code's "Run Python File" command.
        import sys

        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root))
        from src.logger import setup_logging

    setup_logging()
    try:
        main()
    except Exception:
        LOGGER.exception("Data ingestion failed.")
        raise
