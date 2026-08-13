"""Build model features and save reproducible processed datasets.

This module reads a cleaned interim dataset, creates explicitly requested
features, and optionally splits the result into training, validation, and test
sets. It intentionally leaves imputation, scaling, and categorical encoding to
the model pipeline, where those transformations can be fitted on training data
only and data leakage can be avoided.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    # Used when imported as part of the churn_model package.
    from ..data.ingestion import load_dataset
except ImportError:
    # Support VS Code's "Run Python File" command.
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src.churn_model.data.ingestion import load_dataset


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Configuration for feature creation and dataset splitting."""

    source_path: Path
    output_dir: Path = DEFAULT_PROCESSED_DIR
    target_column: str | None = None
    drop_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    keep_date_columns: bool = False
    age_from_year_columns: tuple[str, ...] = ()
    reference_year: int | None = None
    log_columns: tuple[str, ...] = ()
    ratio_features: tuple[str, ...] = ()
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42
    stratify: bool = True
    output_format: str = "csv"
    overwrite: bool = False

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(fraction < 0 for fraction in fractions):
            raise ValueError("Split fractions cannot be negative.")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError(
                "train_fraction, validation_fraction, and test_fraction "
                "must add up to 1.0."
            )
        if self.train_fraction <= 0:
            raise ValueError("train_fraction must be greater than zero.")
        if self.output_format not in {"csv", "parquet"}:
            raise ValueError("output_format must be 'csv' or 'parquet'.")
        if self.age_from_year_columns and self.reference_year is None:
            raise ValueError(
                "reference_year is required when age_from_year_columns "
                "are provided."
            )


@dataclass(frozen=True)
class FeatureBuildReport:
    """Summary of a completed feature-building run."""

    source_path: Path
    output_paths: dict[str, Path]
    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    dropped_target_rows: int = 0
    engineered_features: tuple[str, ...] = ()
    split_rows: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Return a human-readable run summary."""

        lines = [
            "Feature building completed.",
            f"Source: {self.source_path}",
            f"Rows: {self.input_rows} -> {self.output_rows}",
            f"Columns: {self.input_columns} -> {self.output_columns}",
        ]
        if self.dropped_target_rows:
            lines.append(
                "Rows removed because the target was missing: "
                f"{self.dropped_target_rows}"
            )
        if self.engineered_features:
            lines.append(
                f"Engineered features: {list(self.engineered_features)}"
            )
        if self.split_rows:
            lines.append(f"Split sizes: {self.split_rows}")
        lines.extend(
            f"Output ({name}): {path}"
            for name, path in self.output_paths.items()
        )
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    purpose: str,
) -> None:
    """Raise a useful error when requested columns are unavailable."""

    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(
            f"Columns requested for {purpose} were not found: {missing}. "
            "Use the normalized column names from the cleaned dataset."
        )


def _parse_ratio_feature(specification: str) -> tuple[str, str, str]:
    """Parse ``new_feature=numerator/denominator``."""

    try:
        feature_name, expression = specification.split("=", maxsplit=1)
        numerator, denominator = expression.split("/", maxsplit=1)
    except ValueError as error:
        raise ValueError(
            "Ratio features must use "
            "'new_feature=numerator/denominator'. "
            f"Received: {specification!r}"
        ) from error

    parts = tuple(
        part.strip()
        for part in (feature_name, numerator, denominator)
    )
    if any(not part for part in parts):
        raise ValueError(
            "Ratio feature names and columns cannot be empty: "
            f"{specification!r}"
        )
    return parts


def _ensure_new_feature(data: pd.DataFrame, feature_name: str) -> None:
    """Prevent an engineered feature from overwriting an existing column."""

    if feature_name in data.columns:
        raise ValueError(
            f"Engineered feature '{feature_name}' already exists as a column."
        )


def build_features(
    data: pd.DataFrame,
    config: FeatureBuildConfig,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], int]:
    """Create features without modifying the input DataFrame.

    Returns the feature table, engineered feature names, warnings, and the
    number of rows removed because their target value was missing.
    """

    if data.empty:
        raise ValueError("The cleaned dataset contains no rows.")
    if len(data.columns) == 0:
        raise ValueError("The cleaned dataset contains no columns.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    feature_data = data.copy()
    warnings: list[str] = []
    engineered_features: list[str] = []

    if config.target_column is not None:
        _require_columns(feature_data, (config.target_column,), "the target")
        if config.target_column in config.drop_columns:
            raise ValueError("The target column cannot also be dropped.")

    _require_columns(feature_data, config.drop_columns, "dropping")
    if config.drop_columns:
        feature_data = feature_data.drop(columns=list(config.drop_columns))
        LOGGER.info("Dropped columns: %s", list(config.drop_columns))

    _require_columns(feature_data, config.date_columns, "date features")
    for column in config.date_columns:
        original = feature_data[column]
        parsed = pd.to_datetime(original, errors="coerce")
        invalid_count = int((original.notna() & parsed.isna()).sum())
        if invalid_count:
            warnings.append(
                f"'{column}' contained {invalid_count} invalid dates; "
                "their derived features are missing."
            )

        date_features = {
            f"{column}_year": parsed.dt.year,
            f"{column}_month": parsed.dt.month,
            f"{column}_quarter": parsed.dt.quarter,
            f"{column}_day_of_week": parsed.dt.dayofweek,
        }
        for feature_name, values in date_features.items():
            _ensure_new_feature(feature_data, feature_name)
            feature_data[feature_name] = values
            engineered_features.append(feature_name)

        if config.keep_date_columns:
            feature_data[column] = parsed
        else:
            feature_data = feature_data.drop(columns=[column])

    _require_columns(
        feature_data,
        config.age_from_year_columns,
        "age features",
    )
    for column in config.age_from_year_columns:
        year = pd.to_numeric(feature_data[column], errors="coerce")
        feature_name = f"{column}_age"
        _ensure_new_feature(feature_data, feature_name)
        age = config.reference_year - year
        future_year_count = int((age < 0).sum())
        if future_year_count:
            warnings.append(
                f"'{column}' contained {future_year_count} values after "
                f"reference year {config.reference_year}; their ages are missing."
            )
        feature_data[feature_name] = age.mask(age < 0)
        engineered_features.append(feature_name)

    _require_columns(feature_data, config.log_columns, "log features")
    for column in config.log_columns:
        numeric = pd.to_numeric(feature_data[column], errors="coerce")
        invalid_count = int((feature_data[column].notna() & numeric.isna()).sum())
        negative_count = int((numeric < 0).sum())
        if invalid_count:
            warnings.append(
                f"'{column}' contained {invalid_count} non-numeric values; "
                "their log features are missing."
            )
        if negative_count:
            warnings.append(
                f"'{column}' contained {negative_count} negative values; "
                "their log features are missing."
            )

        feature_name = f"{column}_log1p"
        _ensure_new_feature(feature_data, feature_name)
        feature_data[feature_name] = np.log1p(numeric.where(numeric >= 0))
        engineered_features.append(feature_name)

    for specification in config.ratio_features:
        feature_name, numerator, denominator = _parse_ratio_feature(
            specification
        )
        _require_columns(
            feature_data,
            (numerator, denominator),
            f"ratio feature '{feature_name}'",
        )
        _ensure_new_feature(feature_data, feature_name)

        numerator_values = pd.to_numeric(
            feature_data[numerator],
            errors="coerce",
        )
        denominator_values = pd.to_numeric(
            feature_data[denominator],
            errors="coerce",
        )
        zero_count = int((denominator_values == 0).sum())
        if zero_count:
            warnings.append(
                f"Ratio '{feature_name}' had {zero_count} zero denominators; "
                "those results are missing."
            )
        safe_denominator = denominator_values.replace(0, np.nan)
        feature_data[feature_name] = numerator_values / safe_denominator
        engineered_features.append(feature_name)

    dropped_target_rows = 0
    if config.target_column is not None:
        missing_target = feature_data[config.target_column].isna()
        dropped_target_rows = int(missing_target.sum())
        if dropped_target_rows:
            feature_data = feature_data.loc[~missing_target].copy()
            warnings.append(
                f"Removed {dropped_target_rows} rows with a missing target."
            )

    if feature_data.empty:
        raise ValueError("No rows remain after feature building.")

    feature_data = feature_data.reset_index(drop=True)
    return (
        feature_data,
        tuple(engineered_features),
        tuple(warnings),
        dropped_target_rows,
    )


def _allocate_counts(total: int, fractions: tuple[float, ...]) -> list[int]:
    """Allocate every row using the largest-remainder method."""

    exact = np.asarray(fractions, dtype=float) * total
    counts = np.floor(exact).astype(int)
    rows_left = total - int(counts.sum())
    order = np.argsort(-(exact - counts))
    for index in order[:rows_left]:
        counts[index] += 1
    return counts.tolist()


def _can_stratify(target: pd.Series, split_count: int) -> tuple[bool, str | None]:
    """Determine whether a target can be safely stratified."""

    class_counts = target.value_counts(dropna=False)
    unique_count = len(class_counts)
    if unique_count < 2:
        return False, "The target has fewer than two classes; using random split."

    maximum_classes = max(20, int(np.sqrt(len(target))))
    if unique_count > maximum_classes:
        return (
            False,
            "The target appears continuous or has too many classes; "
            "using random split.",
        )

    if int(class_counts.min()) < split_count:
        return (
            False,
            "At least one target class has too few rows for every split; "
            "using random split.",
        )
    return True, None


def split_dataset(
    data: pd.DataFrame,
    config: FeatureBuildConfig,
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Create deterministic train, validation, and test datasets."""

    fractions = (
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    )
    names = ("train", "validation", "test")
    active_split_count = sum(fraction > 0 for fraction in fractions)
    rng = np.random.default_rng(config.random_state)

    use_stratification = False
    stratification_warning: str | None = None
    if config.stratify and config.target_column is not None:
        use_stratification, stratification_warning = _can_stratify(
            data[config.target_column],
            active_split_count,
        )

    split_indices: list[list[int]] = [[], [], []]
    if use_stratification:
        for _, group in data.groupby(
            config.target_column,
            sort=False,
            dropna=False,
        ):
            indices = group.index.to_numpy(copy=True)
            rng.shuffle(indices)
            counts = _allocate_counts(len(indices), fractions)
            start = 0
            for split_index, count in enumerate(counts):
                split_indices[split_index].extend(
                    indices[start : start + count].tolist()
                )
                start += count
    else:
        indices = data.index.to_numpy(copy=True)
        rng.shuffle(indices)
        counts = _allocate_counts(len(indices), fractions)
        start = 0
        for split_index, count in enumerate(counts):
            split_indices[split_index].extend(
                indices[start : start + count].tolist()
            )
            start += count

    datasets: dict[str, pd.DataFrame] = {}
    for name, fraction, indices in zip(names, fractions, split_indices):
        if fraction == 0:
            continue
        if not indices:
            raise ValueError(
                f"The '{name}' split is empty. Use more rows or adjust "
                "the split fractions."
            )
        shuffled_indices = np.asarray(indices)
        rng.shuffle(shuffled_indices)
        datasets[name] = data.loc[shuffled_indices].reset_index(drop=True)

    if sum(len(dataset) for dataset in datasets.values()) != len(data):
        raise RuntimeError("Dataset splitting lost or duplicated rows.")
    return datasets, stratification_warning


def _save_dataframe(data: pd.DataFrame, path: Path) -> None:
    """Save one processed dataset."""

    if path.suffix == ".csv":
        data.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        data.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output path: {path}")


def _json_safe_config(config: FeatureBuildConfig) -> dict[str, Any]:
    """Convert the dataclass configuration into JSON-safe values."""

    values = asdict(config)
    for key in ("source_path", "output_dir"):
        values[key] = str(values[key])
    return values


def build_feature_datasets(config: FeatureBuildConfig) -> FeatureBuildReport:
    """Load interim data, build features, split, save, and report."""

    source = config.source_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    LOGGER.info(
        "Starting feature building: source=%s, output_dir=%s",
        source,
        output_dir,
    )

    if not source.is_file():
        raise FileNotFoundError(f"Cleaned dataset was not found: {source}")

    data = load_dataset(source)
    input_rows = len(data)
    input_columns = len(data.columns)
    feature_data, engineered_features, feature_warnings, dropped_rows = (
        build_features(data, config)
    )

    warnings = list(feature_warnings)
    if config.target_column is None:
        datasets = {"features": feature_data}
    else:
        datasets, split_warning = split_dataset(feature_data, config)
        if split_warning:
            warnings.append(split_warning)

    extension = ".csv" if config.output_format == "csv" else ".parquet"
    output_paths = {
        name: output_dir / f"{name}{extension}"
        for name in datasets
    }
    metadata_path = output_dir / "feature_metadata.json"

    existing_paths = [
        path
        for path in (*output_paths.values(), metadata_path)
        if path.exists()
    ]
    if existing_paths and not config.overwrite:
        locations = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Processed outputs already exist: {locations}. "
            "Use --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataset in datasets.items():
        output_path = output_paths[name]
        _save_dataframe(dataset, output_path)
        LOGGER.info(
            "Saved %s split: rows=%s, columns=%s, path=%s",
            name,
            len(dataset),
            len(dataset.columns),
            output_path,
        )

    metadata = {
        "config": _json_safe_config(config),
        "input_rows": input_rows,
        "output_rows": len(feature_data),
        "columns": list(feature_data.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in feature_data.dtypes.items()
        },
        "engineered_features": list(engineered_features),
        "split_rows": {
            name: len(dataset)
            for name, dataset in datasets.items()
        },
        "warnings": warnings,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved feature metadata to %s", metadata_path)

    for warning in warnings:
        LOGGER.warning("Feature-building warning: %s", warning)

    report = FeatureBuildReport(
        source_path=source,
        output_paths={**output_paths, "metadata": metadata_path},
        input_rows=input_rows,
        output_rows=len(feature_data),
        input_columns=input_columns,
        output_columns=len(feature_data.columns),
        dropped_target_rows=dropped_rows,
        engineered_features=engineered_features,
        split_rows={name: len(dataset) for name, dataset in datasets.items()},
        warnings=tuple(warnings),
    )
    LOGGER.info(
        "Feature building completed: rows=%s, columns=%s, features_created=%s",
        report.output_rows,
        report.output_columns,
        len(report.engineered_features),
    )
    return report


def main() -> None:
    """Run feature building from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Build features from cleaned interim data and save processed data."
        )
    )
    parser.add_argument("source", type=Path, help="Cleaned interim dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed-data directory (default: data/processed)",
    )
    parser.add_argument(
        "--target-column",
        help="Prediction target; enables train/validation/test splitting",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=(),
        help="Columns to remove, such as IDs or names",
    )
    parser.add_argument(
        "--date-columns",
        nargs="*",
        default=(),
        help="Dates from which year, month, quarter, and weekday are created",
    )
    parser.add_argument(
        "--keep-date-columns",
        action="store_true",
        help="Keep original parsed dates after creating date features",
    )
    parser.add_argument(
        "--age-from-year-columns",
        nargs="*",
        default=(),
        help="Year columns from which age features should be created",
    )
    parser.add_argument(
        "--reference-year",
        type=int,
        help="Fixed year used for age features, for example 2026",
    )
    parser.add_argument(
        "--log-columns",
        nargs="*",
        default=(),
        help="Non-negative numeric columns for log1p features",
    )
    parser.add_argument(
        "--ratio-feature",
        action="append",
        default=[],
        help=(
            "Repeatable ratio definition, for example "
            "funding_per_employee=total_funding/employee_count"
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Use a random split even when a categorical target is provided",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        default="csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed datasets and metadata",
    )
    args = parser.parse_args()

    config = FeatureBuildConfig(
        source_path=args.source,
        output_dir=args.output_dir,
        target_column=args.target_column,
        drop_columns=tuple(args.drop_columns),
        date_columns=tuple(args.date_columns),
        keep_date_columns=args.keep_date_columns,
        age_from_year_columns=tuple(args.age_from_year_columns),
        reference_year=args.reference_year,
        log_columns=tuple(args.log_columns),
        ratio_features=tuple(args.ratio_feature),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
        stratify=not args.no_stratify,
        output_format=args.output_format,
        overwrite=args.overwrite,
    )
    report = build_feature_datasets(config)
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
        LOGGER.exception("Feature building failed.")
        raise
