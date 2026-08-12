"""Validate ingested data before cleaning and feature engineering."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
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


class DataValidationError(ValueError):
    """Raised when a dataset does not satisfy the validation rules."""


@dataclass(frozen=True)
class ValidationConfig:
    """Rules used to validate a dataset."""

    data_path: Path
    required_columns: tuple[str, ...] = ()
    target_column: str | None = None
    allowed_target_values: tuple[Any, ...] = ()
    max_missing_fraction: float = 1.0
    fail_on_duplicate_rows: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be between 0.0 and 1.0")


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a successfully validated dataset."""

    data_path: Path
    row_count: int
    column_count: int
    duplicate_row_count: int
    missing_values: dict[str, int]
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Return a readable validation summary."""

        lines = [
            "Data validation passed.",
            f"File: {self.data_path}",
            f"Shape: {self.row_count} rows x {self.column_count} columns",
            f"Duplicate rows: {self.duplicate_row_count}",
        ]

        columns_with_missing_values = {
            column: count
            for column, count in self.missing_values.items()
            if count > 0
        }
        if columns_with_missing_values:
            lines.append(f"Missing values: {columns_with_missing_values}")
        else:
            lines.append("Missing values: none")

        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def validate_dataframe(
    data: pd.DataFrame,
    config: ValidationConfig,
) -> ValidationReport:
    """Validate a DataFrame and return a report if all checks pass."""

    errors: list[str] = []
    warnings: list[str] = []

    if data.empty:
        errors.append("The dataset contains no rows.")

    if len(data.columns) == 0:
        errors.append("The dataset contains no columns.")

    duplicate_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicate_columns:
        errors.append(f"Duplicate column names found: {duplicate_columns}")

    blank_columns = [column for column in data.columns if not str(column).strip()]
    if blank_columns:
        errors.append("One or more columns have blank names.")

    missing_required_columns = sorted(
        set(config.required_columns) - set(data.columns)
    )
    if missing_required_columns:
        errors.append(
            f"Missing required columns: {missing_required_columns}"
        )

    missing_values = {
        str(column): int(count)
        for column, count in data.isna().sum().items()
    }

    if len(data) > 0 and not duplicate_columns:
        excessive_missing_columns = {
            str(column): round(float(data[column].isna().mean()), 4)
            for column in data.columns
            if float(data[column].isna().mean())
            > config.max_missing_fraction
        }
        if excessive_missing_columns:
            errors.append(
                "Columns exceed the maximum missing-value fraction "
                f"({config.max_missing_fraction}): {excessive_missing_columns}"
            )

    duplicate_row_count = int(data.duplicated().sum())
    if duplicate_row_count:
        message = f"Found {duplicate_row_count} duplicate rows."
        if config.fail_on_duplicate_rows:
            errors.append(message)
        else:
            warnings.append(message)

    if config.target_column is not None:
        if config.target_column not in data.columns:
            errors.append(f"Target column not found: {config.target_column}")
        else:
            target = data[config.target_column]
            missing_target_count = int(target.isna().sum())
            if missing_target_count:
                errors.append(
                    f"Target column '{config.target_column}' contains "
                    f"{missing_target_count} missing values."
                )

            if config.allowed_target_values:
                observed_values = set(target.dropna().astype(str).str.strip())
                allowed_values = {
                    str(value).strip()
                    for value in config.allowed_target_values
                }
                unexpected_values = sorted(observed_values - allowed_values)
                if unexpected_values:
                    errors.append(
                        f"Target column '{config.target_column}' contains "
                        f"unexpected values: {unexpected_values}"
                    )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise DataValidationError(f"Data validation failed:\n{details}")

    return ValidationReport(
        data_path=config.data_path,
        row_count=len(data),
        column_count=len(data.columns),
        duplicate_row_count=duplicate_row_count,
        missing_values=missing_values,
        warnings=tuple(warnings),
    )


def validate_data(
    data_path: str | Path,
    *,
    required_columns: tuple[str, ...] = (),
    target_column: str | None = None,
    allowed_target_values: tuple[Any, ...] = (),
    max_missing_fraction: float = 1.0,
    fail_on_duplicate_rows: bool = False,
) -> ValidationReport:
    """Load a dataset, apply validation rules, and return a report."""

    path = Path(data_path).expanduser().resolve()
    LOGGER.info("Starting data validation for %s", path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    config = ValidationConfig(
        data_path=path,
        required_columns=required_columns,
        target_column=target_column,
        allowed_target_values=allowed_target_values,
        max_missing_fraction=max_missing_fraction,
        fail_on_duplicate_rows=fail_on_duplicate_rows,
    )
    data = load_dataset(path)
    report = validate_dataframe(data, config)
    for warning in report.warnings:
        LOGGER.warning("Validation warning for %s: %s", path, warning)

    LOGGER.info(
        "Validated %s rows and %s columns in %s",
        report.row_count,
        report.column_count,
        path,
    )
    return report


def main() -> None:
    """Run data validation from the command line."""

    parser = argparse.ArgumentParser(
        description="Validate an ingested tabular dataset."
    )
    parser.add_argument("data_path", type=Path, help="Dataset to validate")
    parser.add_argument(
        "--required-columns",
        nargs="*",
        default=(),
        help="Column names that must exist",
    )
    parser.add_argument(
        "--target-column",
        help="Optional prediction-target column",
    )
    parser.add_argument(
        "--allowed-target-values",
        nargs="*",
        default=(),
        help="Optional valid values for the target column",
    )
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=1.0,
        help="Maximum missing fraction allowed per column, from 0 to 1",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Treat duplicate rows as a validation failure",
    )
    args = parser.parse_args()

    try:
        report = validate_data(
            args.data_path,
            required_columns=tuple(args.required_columns),
            target_column=args.target_column,
            allowed_target_values=tuple(args.allowed_target_values),
            max_missing_fraction=args.max_missing_fraction,
            fail_on_duplicate_rows=args.fail_on_duplicates,
        )
    except (DataValidationError, FileNotFoundError, ValueError) as error:
        LOGGER.error("Validation failed for %s: %s", args.data_path, error)
        raise SystemExit(1) from error
    except Exception:
        LOGGER.exception("Unexpected validation error for %s", args.data_path)
        raise

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
    main()
