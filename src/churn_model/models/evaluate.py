"""Evaluate a saved classification pipeline on labelled data."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..data.ingestion import load_dataset
    from .train import (
        calculate_classification_metrics,
        load_model_artifact,
        prepare_feature_matrix,
        save_json,
    )
except ImportError:
    # Support VS Code's "Run Python File" command.
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src.churn_model.data.ingestion import load_dataset
    from src.churn_model.models.train import (
        calculate_classification_metrics,
        load_model_artifact,
        prepare_feature_matrix,
        save_json,
    )


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "startup_classifier.joblib"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "evaluation_metrics.json"
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for model evaluation."""

    data_path: Path
    model_path: Path = DEFAULT_MODEL_PATH
    report_path: Path = DEFAULT_REPORT_PATH
    overwrite: bool = False


@dataclass(frozen=True)
class EvaluationReport:
    """Summary of a completed evaluation."""

    data_path: Path
    model_path: Path
    report_path: Path
    evaluated_rows: int
    dropped_target_rows: int
    accuracy: float
    balanced_accuracy: float
    f1_weighted: float

    def summary(self) -> str:
        """Return a readable evaluation summary."""

        return "\n".join(
            [
                "Model evaluation completed.",
                f"Evaluated rows: {self.evaluated_rows}",
                f"Rows skipped because target was missing: "
                f"{self.dropped_target_rows}",
                f"Accuracy: {self.accuracy:.4f}",
                f"Balanced accuracy: {self.balanced_accuracy:.4f}",
                f"Weighted F1: {self.f1_weighted:.4f}",
                f"Model: {self.model_path}",
                f"Report: {self.report_path}",
            ]
        )


def evaluate_model(config: EvaluationConfig) -> EvaluationReport:
    """Load a model, evaluate labelled data, and save detailed metrics."""

    data_path = config.data_path.expanduser().resolve()
    model_path = config.model_path.expanduser().resolve()
    report_path = config.report_path.expanduser().resolve()
    LOGGER.info(
        "Starting model evaluation: model=%s, data=%s",
        model_path,
        data_path,
    )

    if not data_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset was not found: {data_path}")
    if report_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Evaluation report already exists: {report_path}. "
            "Use --overwrite to replace it."
        )

    artifact = load_model_artifact(model_path)
    target_column = artifact["target_column"]
    data = load_dataset(data_path)
    if target_column not in data.columns:
        raise ValueError(
            f"Evaluation data does not contain target column '{target_column}'."
        )

    missing_target = data[target_column].isna()
    dropped_target_rows = int(missing_target.sum())
    if dropped_target_rows:
        LOGGER.warning(
            "Skipping %s evaluation rows with missing targets",
            dropped_target_rows,
        )
        data = data.loc[~missing_target].copy()
    if data.empty:
        raise ValueError("No evaluation rows remain after target checks.")

    X_evaluation = prepare_feature_matrix(
        data,
        artifact["feature_columns"],
        artifact["categorical_columns"],
    )
    y_evaluation = data[target_column]
    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(X_evaluation)
    probabilities = (
        pipeline.predict_proba(X_evaluation)
        if hasattr(pipeline, "predict_proba")
        else None
    )
    metrics = calculate_classification_metrics(
        y_evaluation,
        predictions,
        probabilities,
        artifact["classes"],
    )

    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "target_column": target_column,
        "evaluated_rows": len(data),
        "dropped_target_rows": dropped_target_rows,
        "metrics": metrics,
    }
    save_json(payload, report_path)

    LOGGER.info("Saved evaluation report to %s", report_path)
    LOGGER.info(
        "Model evaluation completed: rows=%s, accuracy=%.4f, "
        "balanced_accuracy=%.4f, weighted_f1=%.4f",
        len(data),
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_weighted"],
    )
    return EvaluationReport(
        data_path=data_path,
        model_path=model_path,
        report_path=report_path,
        evaluated_rows=len(data),
        dropped_target_rows=dropped_target_rows,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        f1_weighted=metrics["f1_weighted"],
    )


def main() -> None:
    """Run model evaluation from the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate a saved classification pipeline."
    )
    parser.add_argument("data", type=Path, help="Labelled test dataset")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved joblib model artifact",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluation report",
    )
    args = parser.parse_args()

    report = evaluate_model(
        EvaluationConfig(
            data_path=args.data,
            model_path=args.model,
            report_path=args.report_output,
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
        LOGGER.exception("Model evaluation failed.")
        raise

