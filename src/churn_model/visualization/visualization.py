"""Generate reusable exploratory charts from cleaned or processed data.

Charts are written to ``reports/figures`` by default. The module uses a
non-interactive Matplotlib backend, so it works from terminals, CI jobs, and
servers without opening chart windows.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ..data.ingestion import load_dataset
except ImportError:
    # Support VS Code's "Run Python File" command.
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src.churn_model.data.ingestion import load_dataset


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True)
class VisualizationConfig:
    """Options controlling chart selection and output."""

    data_path: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    target_column: str | None = None
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    max_numeric_columns: int = 12
    max_categorical_columns: int = 6
    max_categories: int = 12
    histogram_bins: int = 30
    dpi: int = 150
    overwrite: bool = False

    def __post_init__(self) -> None:
        positive_options = {
            "max_numeric_columns": self.max_numeric_columns,
            "max_categorical_columns": self.max_categorical_columns,
            "max_categories": self.max_categories,
            "histogram_bins": self.histogram_bins,
            "dpi": self.dpi,
        }
        invalid = [name for name, value in positive_options.items() if value <= 0]
        if invalid:
            raise ValueError(
                f"These visualization options must be positive: {invalid}"
            )


@dataclass(frozen=True)
class VisualizationReport:
    """Summary of a completed visualization run."""

    data_path: Path
    output_dir: Path
    row_count: int
    column_count: int
    charts: dict[str, Path] = field(default_factory=dict)
    overview_path: Path | None = None
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Return a human-readable run summary."""

        lines = [
            "Visualization completed.",
            f"Dataset: {self.data_path}",
            f"Shape: {self.row_count} rows x {self.column_count} columns",
            f"Output directory: {self.output_dir}",
        ]
        lines.extend(
            f"Chart ({name}): {path}"
            for name, path in self.charts.items()
        )
        if self.overview_path is not None:
            lines.append(f"Dataset overview: {self.overview_path}")
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    purpose: str,
) -> None:
    """Raise an informative error for unknown requested columns."""

    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(
            f"Columns requested for {purpose} were not found: {missing}. "
            "Use column names from the cleaned dataset."
        )


def _select_columns(
    data: pd.DataFrame,
    config: VisualizationConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Select numeric and categorical columns and return any warnings."""

    warnings: list[str] = []
    if config.target_column is not None:
        _require_columns(data, (config.target_column,), "the target chart")

    if config.numeric_columns:
        _require_columns(data, config.numeric_columns, "numeric charts")
        numeric_columns = list(config.numeric_columns)
    else:
        numeric_columns = list(
            data.select_dtypes(include=["number"]).columns
        )
        if config.target_column in numeric_columns:
            numeric_columns.remove(config.target_column)

    if config.categorical_columns:
        _require_columns(
            data,
            config.categorical_columns,
            "categorical charts",
        )
        categorical_columns = list(config.categorical_columns)
    else:
        categorical_candidates = list(
            data.select_dtypes(
                include=["object", "string", "category", "bool"]
            ).columns
        )
        if config.target_column in categorical_candidates:
            categorical_candidates.remove(config.target_column)

        categorical_columns = []
        for column in categorical_candidates:
            unique_count = int(data[column].nunique(dropna=True))
            unique_fraction = unique_count / len(data)
            id_like_name = str(column).lower() == "id" or str(column).lower().endswith(
                "_id"
            )
            high_cardinality = (
                unique_count > config.max_categories * 5
                and unique_fraction > 0.5
            )
            if id_like_name or high_cardinality:
                warnings.append(
                    f"Skipped high-cardinality categorical column '{column}' "
                    f"({unique_count} unique values)."
                )
                continue
            categorical_columns.append(column)

    if len(numeric_columns) > config.max_numeric_columns:
        warnings.append(
            f"Selected the first {config.max_numeric_columns} of "
            f"{len(numeric_columns)} numeric columns."
        )
        numeric_columns = numeric_columns[: config.max_numeric_columns]

    if len(categorical_columns) > config.max_categorical_columns:
        warnings.append(
            f"Selected the first {config.max_categorical_columns} of "
            f"{len(categorical_columns)} categorical columns."
        )
        categorical_columns = categorical_columns[
            : config.max_categorical_columns
        ]

    return numeric_columns, categorical_columns, warnings


def _display_name(column: str) -> str:
    """Convert snake_case column names into readable chart labels."""

    return re.sub(r"_+", " ", str(column)).strip().title()


def _shorten_label(value: Any, maximum_length: int = 28) -> str:
    """Create a readable category label without allowing huge axis text."""

    if pd.isna(value):
        label = "Missing"
    else:
        label = str(value)
    if len(label) > maximum_length:
        return f"{label[: maximum_length - 1]}…"
    return label


def _category_counts(series: pd.Series, limit: int) -> pd.Series:
    """Return top category counts and combine the remainder as Other."""

    counts = series.value_counts(dropna=False)
    if len(counts) > limit:
        displayed = counts.iloc[:limit].copy()
        displayed.loc["Other"] = int(counts.iloc[limit:].sum())
        counts = displayed
    counts.index = [_shorten_label(value) for value in counts.index]
    return counts


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Save and close one Matplotlib figure."""

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_target(
    data: pd.DataFrame,
    column: str,
    path: Path,
    config: VisualizationConfig,
) -> None:
    """Create a target-frequency chart."""

    counts = _category_counts(data[column], config.max_categories)
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(
        counts.index,
        counts.values,
        color="#2563EB",
        edgecolor="white",
    )
    axis.set_title(f"Target Distribution: {_display_name(column)}")
    axis.set_xlabel(_display_name(column))
    axis.set_ylabel("Number of Rows")
    axis.tick_params(axis="x", rotation=35)
    axis.bar_label(bars, padding=3, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    _save_figure(fig, path, config.dpi)


def _plot_missing_values(
    data: pd.DataFrame,
    path: Path,
    config: VisualizationConfig,
) -> None:
    """Create a horizontal missing-value chart."""

    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    missing_percent = missing / len(data) * 100

    height = max(4.5, min(12, 0.45 * len(missing) + 2))
    fig, axis = plt.subplots(figsize=(11, height))
    bars = axis.barh(
        [_display_name(column) for column in missing.index],
        missing_percent.values,
        color="#F59E0B",
    )
    axis.set_title("Missing Values by Column")
    axis.set_xlabel("Missing Values (%)")
    axis.set_ylabel("Column")
    axis.grid(axis="x", alpha=0.2)
    axis.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in missing_percent.values],
        padding=3,
        fontsize=8,
    )
    _save_figure(fig, path, config.dpi)


def _subplot_grid(item_count: int, maximum_columns: int = 3) -> tuple[int, int]:
    """Return a compact row and column count for a chart grid."""

    columns = min(maximum_columns, item_count)
    rows = math.ceil(item_count / columns)
    return rows, columns


def _plot_numeric_distributions(
    data: pd.DataFrame,
    columns: list[str],
    path: Path,
    config: VisualizationConfig,
) -> None:
    """Create histograms for selected numeric features."""

    rows, grid_columns = _subplot_grid(len(columns))
    fig, axes = plt.subplots(
        rows,
        grid_columns,
        figsize=(5 * grid_columns, 3.8 * rows),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, columns):
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            axis.text(
                0.5,
                0.5,
                "No numeric values",
                ha="center",
                va="center",
            )
        else:
            axis.hist(
                values,
                bins=config.histogram_bins,
                color="#0EA5E9",
                edgecolor="white",
            )
            median = float(values.median())
            axis.axvline(
                median,
                color="#DC2626",
                linestyle="--",
                linewidth=1.5,
                label=f"Median: {median:,.2f}",
            )
            axis.legend(fontsize=8)
        axis.set_title(_display_name(column))
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")
        axis.grid(axis="y", alpha=0.15)

    for axis in list(axes.flat)[len(columns) :]:
        axis.set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=16, y=1.01)
    _save_figure(fig, path, config.dpi)


def _plot_correlations(
    data: pd.DataFrame,
    columns: list[str],
    path: Path,
    config: VisualizationConfig,
) -> None:
    """Create a Pearson-correlation heatmap."""

    numeric = data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr()
    size = max(7, min(15, 0.85 * len(columns) + 3))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(
        correlation,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    labels = [_display_name(column) for column in correlation.columns]
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title("Numeric Feature Correlations")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Correlation")

    if len(columns) <= 10:
        for row in range(len(columns)):
            for column in range(len(columns)):
                value = correlation.iloc[row, column]
                if pd.notna(value):
                    text_color = "white" if abs(value) > 0.55 else "black"
                    axis.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )
    _save_figure(fig, path, config.dpi)


def _plot_categorical_distributions(
    data: pd.DataFrame,
    columns: list[str],
    path: Path,
    config: VisualizationConfig,
) -> None:
    """Create top-category frequency charts."""

    rows, grid_columns = _subplot_grid(len(columns), maximum_columns=2)
    fig, axes = plt.subplots(
        rows,
        grid_columns,
        figsize=(7 * grid_columns, 4.5 * rows),
        squeeze=False,
    )

    for axis, column in zip(axes.flat, columns):
        counts = _category_counts(data[column], config.max_categories)
        counts = counts.sort_values(ascending=True)
        axis.barh(counts.index, counts.values, color="#10B981")
        axis.set_title(_display_name(column))
        axis.set_xlabel("Number of Rows")
        axis.set_ylabel("Category")
        axis.grid(axis="x", alpha=0.15)

    for axis in list(axes.flat)[len(columns) :]:
        axis.set_visible(False)
    fig.suptitle("Categorical Feature Distributions", fontsize=16, y=1.01)
    _save_figure(fig, path, config.dpi)


def _write_overview(
    data: pd.DataFrame,
    data_path: Path,
    numeric_columns: list[str],
    categorical_columns: list[str],
    charts: dict[str, Path],
    warnings: list[str],
    output_path: Path,
) -> None:
    """Write machine-readable dataset and chart metadata."""

    overview = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path),
        "row_count": len(data),
        "column_count": len(data.columns),
        "columns": list(data.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in data.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in data.isna().sum().items()
        },
        "numeric_columns_visualized": numeric_columns,
        "categorical_columns_visualized": categorical_columns,
        "charts": {name: str(path) for name, path in charts.items()},
        "warnings": warnings,
    }
    output_path.write_text(
        json.dumps(overview, indent=2),
        encoding="utf-8",
    )


def generate_visualizations(
    config: VisualizationConfig,
) -> VisualizationReport:
    """Load a dataset, generate applicable charts, and save an overview."""

    data_path = config.data_path.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    LOGGER.info(
        "Starting visualization: data=%s, output_dir=%s",
        data_path,
        output_dir,
    )

    if not data_path.is_file():
        raise FileNotFoundError(f"Visualization dataset was not found: {data_path}")

    data = load_dataset(data_path)
    if data.empty:
        raise ValueError("Visualization dataset contains no rows.")
    if len(data.columns) == 0:
        raise ValueError("Visualization dataset contains no columns.")

    numeric_columns, categorical_columns, warnings = _select_columns(
        data,
        config,
    )
    chart_paths: dict[str, Path] = {}
    if config.target_column is not None:
        chart_paths["target_distribution"] = (
            output_dir / "target_distribution.png"
        )
    if data.isna().any().any():
        chart_paths["missing_values"] = output_dir / "missing_values.png"
    else:
        LOGGER.info("No missing values found; skipping the missing-values chart")
    if numeric_columns:
        chart_paths["numeric_distributions"] = (
            output_dir / "numeric_distributions.png"
        )
    else:
        warnings.append("No numeric columns were available for visualization.")
    if len(numeric_columns) >= 2:
        chart_paths["correlation_heatmap"] = (
            output_dir / "correlation_heatmap.png"
        )
    if categorical_columns:
        chart_paths["categorical_distributions"] = (
            output_dir / "categorical_distributions.png"
        )
    else:
        warnings.append(
            "No categorical columns were available for visualization."
        )

    overview_path = output_dir / "dataset_overview.json"
    existing_outputs = [
        path
        for path in (*chart_paths.values(), overview_path)
        if path.exists()
    ]
    if existing_outputs and not config.overwrite:
        raise FileExistsError(
            "Visualization outputs already exist: "
            f"{', '.join(str(path) for path in existing_outputs)}. "
            "Use --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.target_column is not None:
        _plot_target(
            data,
            config.target_column,
            chart_paths["target_distribution"],
            config,
        )
    if "missing_values" in chart_paths:
        _plot_missing_values(data, chart_paths["missing_values"], config)
    if numeric_columns:
        _plot_numeric_distributions(
            data,
            numeric_columns,
            chart_paths["numeric_distributions"],
            config,
        )
    if len(numeric_columns) >= 2:
        _plot_correlations(
            data,
            numeric_columns,
            chart_paths["correlation_heatmap"],
            config,
        )
    if categorical_columns:
        _plot_categorical_distributions(
            data,
            categorical_columns,
            chart_paths["categorical_distributions"],
            config,
        )

    for name, path in chart_paths.items():
        LOGGER.info("Saved %s chart to %s", name, path)
    for warning in warnings:
        LOGGER.warning("Visualization warning: %s", warning)

    _write_overview(
        data,
        data_path,
        numeric_columns,
        categorical_columns,
        chart_paths,
        warnings,
        overview_path,
    )
    LOGGER.info("Saved dataset overview to %s", overview_path)
    LOGGER.info(
        "Visualization completed: rows=%s, columns=%s, charts=%s",
        len(data),
        len(data.columns),
        len(chart_paths),
    )
    return VisualizationReport(
        data_path=data_path,
        output_dir=output_dir,
        row_count=len(data),
        column_count=len(data.columns),
        charts=chart_paths,
        overview_path=overview_path,
        warnings=tuple(warnings),
    )


def main() -> None:
    """Run visualization generation from the command line."""

    parser = argparse.ArgumentParser(
        description="Generate exploratory charts from a tabular dataset."
    )
    parser.add_argument("data", type=Path, help="Cleaned or processed dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Chart directory (default: reports/figures)",
    )
    parser.add_argument(
        "--target-column",
        help="Optional prediction target for a target-distribution chart",
    )
    parser.add_argument(
        "--numeric-columns",
        nargs="*",
        default=(),
        help="Numeric columns to visualize; defaults to automatic detection",
    )
    parser.add_argument(
        "--categorical-columns",
        nargs="*",
        default=(),
        help="Categorical columns; defaults to automatic detection",
    )
    parser.add_argument("--max-numeric-columns", type=int, default=12)
    parser.add_argument("--max-categorical-columns", type=int, default=6)
    parser.add_argument("--max-categories", type=int, default=12)
    parser.add_argument("--histogram-bins", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing visualization outputs",
    )
    args = parser.parse_args()

    report = generate_visualizations(
        VisualizationConfig(
            data_path=args.data,
            output_dir=args.output_dir,
            target_column=args.target_column,
            numeric_columns=tuple(args.numeric_columns),
            categorical_columns=tuple(args.categorical_columns),
            max_numeric_columns=args.max_numeric_columns,
            max_categorical_columns=args.max_categorical_columns,
            max_categories=args.max_categories,
            histogram_bins=args.histogram_bins,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    )
    print(report.summary())


if __name__ == "__main__":
    try:
        from src.logger import setup_logging
    except ModuleNotFoundError:
        import sys

        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root))
        from src.logger import setup_logging

    setup_logging()
    try:
        main()
    except Exception:
        LOGGER.exception("Visualization failed.")
        raise
