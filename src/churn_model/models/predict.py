"""Generate predictions from a saved classification pipeline."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from ..data.ingestion import load_dataset
    from .train import load_model_artifact, prepare_feature_matrix
except ImportError:
    # Support VS Code's "Run Python File" command.
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src.churn_model.data.ingestion import load_dataset
    from src.churn_model.models.train import (
        load_model_artifact,
        prepare_feature_matrix,
    )


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "startup_classifier.joblib"
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "predictions" / "predictions.csv"
)


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for batch prediction."""

    data_path: Path
    model_path: Path = DEFAULT_MODEL_PATH
    output_path: Path = DEFAULT_PREDICTIONS_PATH
    prediction_column: str | None = None
    include_input: bool = True
    include_probabilities: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class PredictionReport:
    """Summary of a completed batch-prediction run."""

    data_path: Path
    model_path: Path
    output_path: Path
    predicted_rows: int
    prediction_column: str
    probability_columns: tuple[str, ...]

    def summary(self) -> str:
        """Return a readable prediction summary."""

        lines = [
            "Prediction completed.",
            f"Predicted rows: {self.predicted_rows}",
            f"Prediction column: {self.prediction_column}",
        ]
        if self.probability_columns:
            lines.append(
                f"Probability columns: {list(self.probability_columns)}"
            )
        lines.extend(
            [
                f"Model: {self.model_path}",
                f"Output: {self.output_path}",
            ]
        )
        return "\n".join(lines)


def _safe_label(value: object) -> str:
    """Convert a class label into a safe column-name suffix."""

    label = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return label.strip("_") or "unknown"


def _save_predictions(data: pd.DataFrame, path: Path) -> None:
    """Save predictions as CSV or Parquet."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        data.to_csv(path, index=False)
    elif suffix == ".parquet":
        data.to_parquet(path, index=False)
    else:
        raise ValueError(
            "Prediction output must end with '.csv' or '.parquet'."
        )


def predict_data(config: PredictionConfig) -> PredictionReport:
    """Load input data, generate predictions, and save the result."""

    data_path = config.data_path.expanduser().resolve()
    model_path = config.model_path.expanduser().resolve()
    output_path = config.output_path.expanduser().resolve()
    LOGGER.info(
        "Starting prediction: model=%s, data=%s, output=%s",
        model_path,
        data_path,
        output_path,
    )

    if not data_path.is_file():
        raise FileNotFoundError(f"Prediction dataset was not found: {data_path}")
    if output_path == data_path:
        raise ValueError("Prediction output cannot overwrite the input dataset.")
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Prediction output already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    artifact = load_model_artifact(model_path)
    data = load_dataset(data_path)
    if data.empty:
        raise ValueError("Prediction dataset contains no rows.")

    features = prepare_feature_matrix(
        data,
        artifact["feature_columns"],
        artifact["categorical_columns"],
    )
    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(features)
    prediction_column = config.prediction_column or (
        f"predicted_{artifact['target_column']}"
    )

    result = data.copy() if config.include_input else pd.DataFrame(
        {"source_row": range(len(data))}
    )
    if prediction_column in result.columns:
        raise ValueError(
            f"Prediction column '{prediction_column}' already exists in input."
        )
    result[prediction_column] = predictions

    probability_columns: list[str] = []
    if config.include_probabilities and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        used_names: set[str] = set(result.columns)
        for index, class_label in enumerate(artifact["classes"]):
            base_name = f"probability_{_safe_label(class_label)}"
            column_name = base_name
            suffix = 2
            while column_name in used_names:
                column_name = f"{base_name}_{suffix}"
                suffix += 1
            result[column_name] = probabilities[:, index]
            probability_columns.append(column_name)
            used_names.add(column_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_predictions(result, output_path)
    LOGGER.info(
        "Prediction completed: rows=%s, output=%s",
        len(result),
        output_path,
    )
    if probability_columns:
        LOGGER.info("Saved probability columns: %s", probability_columns)

    return PredictionReport(
        data_path=data_path,
        model_path=model_path,
        output_path=output_path,
        predicted_rows=len(result),
        prediction_column=prediction_column,
        probability_columns=tuple(probability_columns),
    )


def main() -> None:
    """Run batch prediction from the command line."""

    parser = argparse.ArgumentParser(
        description="Generate predictions using a saved model pipeline."
    )
    parser.add_argument("data", type=Path, help="Data to predict")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved joblib model artifact",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--prediction-column",
        help="Custom prediction-column name",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Do not copy original input columns to the output",
    )
    parser.add_argument(
        "--no-probabilities",
        action="store_true",
        help="Do not save class-probability columns",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prediction file",
    )
    args = parser.parse_args()

    report = predict_data(
        PredictionConfig(
            data_path=args.data,
            model_path=args.model,
            output_path=args.output,
            prediction_column=args.prediction_column,
            include_input=not args.predictions_only,
            include_probabilities=not args.no_probabilities,
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
        LOGGER.exception("Prediction failed.")
        raise
