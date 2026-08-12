"""Clean raw tabular data and save it in the interim-data directory.

The cleaner performs safe, repeatable operations. It does not fill missing
values, encode categories, create model features, or modify the raw file.
Those decisions belong in later transformation and modelling stages.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

try:
    # Used when imported as part of the churn_model package.
    from .ingestion import load_dataset
except ImportError:
    # Used when this file is executed directly.
    from ingestion import load_dataset


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_MISSING_MARKERS = ("", "n/a", "null", "none", "nan")


@dataclass(frozen=True)
class CleaningConfig:
    """Options used by the cleaning step."""

    source_path: Path
    output_path: Path | None = None
    numeric_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    missing_markers: tuple[str, ...] = DEFAULT_MISSING_MARKERS
    drop_duplicate_rows: bool = True
    drop_empty_rows: bool = True
    overwrite: bool = False

    @property
    def destination_path(self) -> Path:
        """Return the configured or automatically generated output path."""

        if self.output_path is not None:
            return self.output_path

        filename = f"{self.source_path.stem}_cleaned.csv"
        return DEFAULT_INTERIM_DIR / filename


@dataclass(frozen=True)
class CleaningReport:
    """Summary of the changes made during cleaning."""

    source_path: Path
    output_path: Path
    input_rows: int
    output_rows: int
    duplicate_rows_removed: int
    empty_rows_removed: int
    renamed_columns: dict[str, str] = field(default_factory=dict)
    invalid_numeric_values: dict[str, int] = field(default_factory=dict)
    invalid_date_values: dict[str, int] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""

        lines = [
            "Data cleaning completed.",
            f"Source: {self.source_path}",
            f"Output: {self.output_path}",
            f"Rows: {self.input_rows} -> {self.output_rows}",
            f"Empty rows removed: {self.empty_rows_removed}",
            f"Duplicate rows removed: {self.duplicate_rows_removed}",
        ]

        if self.renamed_columns:
            lines.append(f"Renamed columns: {self.renamed_columns}")
        if self.invalid_numeric_values:
            lines.append(
                "Invalid numeric values converted to missing: "
                f"{self.invalid_numeric_values}"
            )
        if self.invalid_date_values:
            lines.append(
                "Invalid date values converted to missing: "
                f"{self.invalid_date_values}"
            )

        columns_with_missing_values = {
            column: count
            for column, count in self.missing_values.items()
            if count > 0
        }
        lines.append(
            "Remaining missing values: "
            f"{columns_with_missing_values or 'none'}"
        )
        return "\n".join(lines)


def normalize_column_name(column: Any) -> str:
    """Convert a column name to lowercase snake_case."""

    name = str(column).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _standardize_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize column names and reject names that become ambiguous."""

    renamed_columns = {
        str(column): normalize_column_name(column)
        for column in data.columns
    }

    empty_names = [old for old, new in renamed_columns.items() if not new]
    if empty_names:
        raise ValueError(
            f"These columns become empty after normalization: {empty_names}"
        )

    normalized_names = list(renamed_columns.values())
    duplicates = sorted(
        {
            name
            for name in normalized_names
            if normalized_names.count(name) > 1
        }
    )
    if duplicates:
        raise ValueError(
            "Column names collide after normalization: "
            f"{duplicates}. Rename them in the source schema."
        )

    changed_names = {
        old: new
        for old, new in renamed_columns.items()
        if old != new
    }
    return data.rename(columns=renamed_columns), changed_names


def _clean_text_values(
    data: pd.DataFrame,
    missing_markers: tuple[str, ...],
) -> pd.DataFrame:
    """Trim text and replace configured missing markers with ``pd.NA``."""

    cleaned = data.copy()
    normalized_markers = {
        marker.strip().casefold()
        for marker in missing_markers
    }

    def clean_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if stripped.casefold() in normalized_markers:
            return pd.NA
        return stripped

    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].map(clean_value)

    return cleaned


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    purpose: str,
) -> None:
    """Raise an informative error when requested columns do not exist."""

    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(
            f"Columns requested for {purpose} were not found: {missing}. "
            "Use normalized snake_case column names."
        )


def _convert_numeric_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert explicitly selected columns to numeric values."""

    _require_columns(data, columns, "numeric conversion")
    cleaned = data.copy()
    invalid_counts: dict[str, int] = {}

    for column in columns:
        original = cleaned[column]
        prepared = original
        if pd.api.types.is_object_dtype(original) or isinstance(
            original.dtype, pd.StringDtype
        ):
            prepared = (
                original.astype("string")
                .str.replace(r"[,\$€£₹]", "", regex=True)
                .str.strip()
            )

        converted = pd.to_numeric(prepared, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        if invalid_count:
            invalid_counts[column] = invalid_count
        cleaned[column] = converted

    return cleaned, invalid_counts


def _convert_date_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert explicitly selected columns to pandas datetime values."""

    _require_columns(data, columns, "date conversion")
    cleaned = data.copy()
    invalid_counts: dict[str, int] = {}

    for column in columns:
        original = cleaned[column]
        converted = pd.to_datetime(original, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        if invalid_count:
            invalid_counts[column] = invalid_count
        cleaned[column] = converted

    return cleaned, invalid_counts


def clean_dataframe(
    data: pd.DataFrame,
    config: CleaningConfig,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply cleaning rules without modifying the input DataFrame."""

    input_rows = len(data)
    cleaned, renamed_columns = _standardize_columns(data.copy())
    cleaned = _clean_text_values(cleaned, config.missing_markers)

    rows_before_empty_removal = len(cleaned)
    if config.drop_empty_rows:
        cleaned = cleaned.dropna(how="all")
    empty_rows_removed = rows_before_empty_removal - len(cleaned)

    rows_before_deduplication = len(cleaned)
    if config.drop_duplicate_rows:
        cleaned = cleaned.drop_duplicates()
    duplicate_rows_removed = rows_before_deduplication - len(cleaned)

    cleaned, invalid_numeric_values = _convert_numeric_columns(
        cleaned,
        config.numeric_columns,
    )
    cleaned, invalid_date_values = _convert_date_columns(
        cleaned,
        config.date_columns,
    )

    cleaned = cleaned.reset_index(drop=True)
    missing_values = {
        str(column): int(count)
        for column, count in cleaned.isna().sum().items()
    }

    report = CleaningReport(
        source_path=config.source_path,
        output_path=config.destination_path,
        input_rows=input_rows,
        output_rows=len(cleaned),
        duplicate_rows_removed=duplicate_rows_removed,
        empty_rows_removed=empty_rows_removed,
        renamed_columns=renamed_columns,
        invalid_numeric_values=invalid_numeric_values,
        invalid_date_values=invalid_date_values,
        missing_values=missing_values,
    )
    return cleaned, report


def save_dataset(data: pd.DataFrame, path: Path) -> None:
    """Save a cleaned dataset using the destination file extension."""

    writers = {
        ".csv": lambda: data.to_csv(path, index=False),
        ".json": lambda: data.to_json(
            path,
            orient="records",
            lines=True,
            date_format="iso",
        ),
        ".parquet": lambda: data.to_parquet(path, index=False),
        ".xlsx": lambda: data.to_excel(path, index=False),
    }

    suffix = path.suffix.lower()
    if suffix not in writers:
        supported = ", ".join(sorted(writers))
        raise ValueError(
            f"Unsupported output type '{suffix}'. Supported types: {supported}"
        )
    writers[suffix]()


def clean_data(config: CleaningConfig) -> CleaningReport:
    """Load raw data, clean it, save interim data, and return a report."""

    source = config.source_path.expanduser().resolve()
    destination = config.destination_path.expanduser().resolve()

    LOGGER.info(
        "Starting data cleaning: source=%s, destination=%s",
        source,
        destination,
    )

    if not source.is_file():
        raise FileNotFoundError(f"Raw dataset was not found: {source}")
    if source == destination:
        raise ValueError("The output path must be different from the raw file.")
    if destination.exists() and not config.overwrite:
        raise FileExistsError(
            f"Cleaned data already exists at {destination}. "
            "Use --overwrite to replace it."
        )

    data = load_dataset(source)
    cleaned, report = clean_dataframe(data, config)

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(cleaned, destination)

    for column, count in report.invalid_numeric_values.items():
        LOGGER.warning(
            "Converted %s invalid numeric values to missing in '%s'",
            count,
            column,
        )
    for column, count in report.invalid_date_values.items():
        LOGGER.warning(
            "Converted %s invalid date values to missing in '%s'",
            count,
            column,
        )

    LOGGER.info(
        "Data cleaning completed: rows=%s, columns=%s, output=%s",
        len(cleaned),
        len(cleaned.columns),
        destination,
    )
    return report


def main() -> None:
    """Run data cleaning from the command line."""

    parser = argparse.ArgumentParser(
        description="Clean raw tabular data and save an interim dataset."
    )
    parser.add_argument("source", type=Path, help="Raw dataset to clean")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output file. Defaults to "
            "data/interim/<source_name>_cleaned.csv"
        ),
    )
    parser.add_argument(
        "--numeric-columns",
        nargs="*",
        default=(),
        help="Columns to convert to numbers (use normalized names)",
    )
    parser.add_argument(
        "--date-columns",
        nargs="*",
        default=(),
        help="Columns to convert to dates (use normalized names)",
    )
    parser.add_argument(
        "--missing-markers",
        nargs="*",
        default=DEFAULT_MISSING_MARKERS,
        help="Text values that should become missing values",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate rows instead of removing them",
    )
    parser.add_argument(
        "--keep-empty-rows",
        action="store_true",
        help="Keep completely empty rows instead of removing them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing interim file",
    )
    args = parser.parse_args()

    config = CleaningConfig(
        source_path=args.source,
        output_path=args.output,
        numeric_columns=tuple(args.numeric_columns),
        date_columns=tuple(args.date_columns),
        missing_markers=tuple(args.missing_markers),
        drop_duplicate_rows=not args.keep_duplicates,
        drop_empty_rows=not args.keep_empty_rows,
        overwrite=args.overwrite,
    )

    report = clean_data(config)
    print(report.summary())


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
        LOGGER.exception("Data cleaning failed.")
        raise
