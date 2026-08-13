"""Train and persist a classification pipeline.

The saved artifact contains both preprocessing and the estimator. Numeric
values are median-imputed and scaled; categorical values are most-frequent
imputed and one-hot encoded. Keeping these operations inside the fitted
pipeline prevents training/validation leakage and makes prediction consistent.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "startup_classifier.joblib"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "training_metrics.json"
)
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for model training."""

    train_path: Path
    target_column: str
    validation_path: Path | None = None
    model_output: Path = DEFAULT_MODEL_PATH
    metrics_output: Path = DEFAULT_METRICS_PATH
    model_type: str = "logistic_regression"
    random_state: int = 42
    max_iter: int = 2_000
    n_estimators: int = 300
    class_weight: str | None = "balanced"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.target_column.strip():
            raise ValueError("target_column cannot be empty.")
        if self.model_type not in {"logistic_regression", "random_forest"}:
            raise ValueError(
                "model_type must be 'logistic_regression' or 'random_forest'."
            )
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a completed training run."""

    model_path: Path
    metrics_path: Path
    train_rows: int
    validation_rows: int
    feature_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    classes: tuple[Any, ...]
    evaluation_dataset: str
    accuracy: float
    f1_weighted: float

    def summary(self) -> str:
        """Return a readable training summary."""

        return "\n".join(
            [
                "Model training completed.",
                f"Training rows: {self.train_rows}",
                f"Validation rows: {self.validation_rows}",
                f"Features: {self.feature_count}",
                f"Numeric features: {self.numeric_feature_count}",
                f"Categorical features: {self.categorical_feature_count}",
                f"Classes: {list(self.classes)}",
                f"Metrics dataset: {self.evaluation_dataset}",
                f"Accuracy: {self.accuracy:.4f}",
                f"Weighted F1: {self.f1_weighted:.4f}",
                f"Model: {self.model_path}",
                f"Metrics: {self.metrics_path}",
            ]
        )


def _json_safe(value: Any) -> Any:
    """Convert paths and NumPy values into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON report, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a model artifact."""

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    required_keys = {
        "artifact_version",
        "pipeline",
        "target_column",
        "feature_columns",
        "numeric_columns",
        "categorical_columns",
        "classes",
    }
    if not isinstance(artifact, dict):
        raise ValueError("The model artifact must contain a dictionary.")
    missing_keys = sorted(required_keys - set(artifact))
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required fields: {missing_keys}"
        )
    if artifact["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported model artifact version: "
            f"{artifact['artifact_version']}"
        )
    return artifact


def _remove_missing_targets(
    data: pd.DataFrame,
    target_column: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Validate the target and remove rows where it is missing."""

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in {dataset_name}."
        )

    missing_target = data[target_column].isna()
    dropped_rows = int(missing_target.sum())
    if dropped_rows:
        LOGGER.warning(
            "Removing %s rows with missing targets from %s",
            dropped_rows,
            dataset_name,
        )
        data = data.loc[~missing_target].copy()

    if data.empty:
        raise ValueError(f"No rows remain in {dataset_name} after target checks.")
    return data, data[target_column], dropped_rows


def prepare_feature_matrix(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    categorical_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Align inference data to the schema used during training."""

    expected = list(feature_columns)
    missing_columns = sorted(set(expected) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing model features: {missing_columns}"
        )

    extra_columns = sorted(set(data.columns) - set(expected))
    if extra_columns:
        LOGGER.debug("Ignoring columns not used by the model: %s", extra_columns)

    features = data.loc[:, expected].copy()
    for column in categorical_columns:
        # Older scikit-learn versions handle np.nan more reliably than pd.NA.
        features[column] = (
            features[column]
            .astype("object")
            .where(features[column].notna(), np.nan)
        )
    return features


def build_pipeline(
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    config: TrainConfig,
) -> Pipeline:
    """Create a leakage-safe preprocessing and classification pipeline."""

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_columns))
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    if config.model_type == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            random_state=config.random_state,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=config.n_estimators,
            class_weight=config.class_weight,
            random_state=config.random_state,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def calculate_classification_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    classes: list[Any] | tuple[Any, ...] | np.ndarray,
) -> dict[str, Any]:
    """Calculate metrics shared by training and standalone evaluation."""

    labels = list(classes)
    observed_labels = set(pd.unique(y_true))
    unknown_labels = observed_labels - set(labels)
    if unknown_labels:
        raise ValueError(
            "Evaluation data contains labels not seen during training: "
            f"{sorted(str(label) for label in unknown_labels)}"
        )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=labels,
            target_names=[str(label) for label in labels],
            output_dict=True,
            zero_division=0,
        ),
    }
    if probabilities is not None:
        metrics["log_loss"] = float(
            log_loss(y_true, probabilities, labels=labels)
        )
    return metrics


def _resolve_validation_path(config: TrainConfig, train_path: Path) -> Path | None:
    """Use the provided validation file or discover its sibling."""

    if config.validation_path is not None:
        return config.validation_path.expanduser().resolve()

    candidate = train_path.with_name(f"validation{train_path.suffix}")
    if candidate.is_file():
        LOGGER.info("Automatically selected validation dataset: %s", candidate)
        return candidate
    return None


def train_model(config: TrainConfig) -> TrainingReport:
    """Train, evaluate, and save a complete classification pipeline."""

    train_path = config.train_path.expanduser().resolve()
    model_output = config.model_output.expanduser().resolve()
    metrics_output = config.metrics_output.expanduser().resolve()
    LOGGER.info(
        "Starting model training: train=%s, target=%s, model_type=%s",
        train_path,
        config.target_column,
        config.model_type,
    )

    if not train_path.is_file():
        raise FileNotFoundError(f"Training dataset was not found: {train_path}")
    existing_outputs = [
        path for path in (model_output, metrics_output) if path.exists()
    ]
    if existing_outputs and not config.overwrite:
        raise FileExistsError(
            "Training outputs already exist: "
            f"{', '.join(str(path) for path in existing_outputs)}. "
            "Use --overwrite to replace them."
        )

    train_data = load_dataset(train_path)
    train_data, y_train, _ = _remove_missing_targets(
        train_data,
        config.target_column,
        "training data",
    )
    feature_columns = tuple(
        column
        for column in train_data.columns
        if column != config.target_column
    )
    if not feature_columns:
        raise ValueError("Training data contains no feature columns.")

    numeric_columns = tuple(
        train_data.loc[:, feature_columns]
        .select_dtypes(include=["number", "bool"])
        .columns
    )
    categorical_columns = tuple(
        column for column in feature_columns if column not in numeric_columns
    )
    all_missing_columns = [
        column for column in feature_columns if train_data[column].isna().all()
    ]
    if all_missing_columns:
        raise ValueError(
            "These training features are completely missing and must be "
            f"removed: {all_missing_columns}"
        )

    for column in categorical_columns:
        unique_count = int(train_data[column].nunique(dropna=True))
        if unique_count > 100 and unique_count / len(train_data) > 0.5:
            LOGGER.warning(
                "Categorical feature '%s' has high cardinality (%s values). "
                "Consider dropping identifiers before training.",
                column,
                unique_count,
            )

    X_train = prepare_feature_matrix(
        train_data,
        feature_columns,
        categorical_columns,
    )
    classes = tuple(pd.unique(y_train))
    if len(classes) < 2:
        raise ValueError("Training target must contain at least two classes.")

    pipeline = build_pipeline(
        numeric_columns,
        categorical_columns,
        config,
    )
    LOGGER.info(
        "Fitting model with %s rows, %s numeric features, and %s categorical "
        "features",
        len(X_train),
        len(numeric_columns),
        len(categorical_columns),
    )
    pipeline.fit(X_train, y_train)
    fitted_classes = tuple(pipeline.named_steps["model"].classes_)

    validation_path = _resolve_validation_path(config, train_path)
    if validation_path is not None:
        if not validation_path.is_file():
            raise FileNotFoundError(
                f"Validation dataset was not found: {validation_path}"
            )
        evaluation_data = load_dataset(validation_path)
        evaluation_data, y_evaluation, _ = _remove_missing_targets(
            evaluation_data,
            config.target_column,
            "validation data",
        )
        evaluation_name = "validation"
    else:
        LOGGER.warning(
            "No validation dataset was supplied or discovered; metrics will "
            "be calculated on training data and may be optimistic."
        )
        evaluation_data = train_data
        y_evaluation = y_train
        evaluation_name = "training"

    X_evaluation = prepare_feature_matrix(
        evaluation_data,
        feature_columns,
        categorical_columns,
    )
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
        fitted_classes,
    )

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "target_column": config.target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "classes": fitted_classes,
        "model_type": config.model_type,
        "random_state": config.random_state,
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_output, compress=3)

    metrics_payload = {
        "model_path": str(model_output),
        "model_type": config.model_type,
        "target_column": config.target_column,
        "training_rows": len(train_data),
        "evaluation_rows": len(evaluation_data),
        "evaluation_dataset": evaluation_name,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "metrics": metrics,
    }
    save_json(metrics_payload, metrics_output)

    LOGGER.info("Saved trained model to %s", model_output)
    LOGGER.info("Saved training metrics to %s", metrics_output)
    LOGGER.info(
        "Model training completed: accuracy=%.4f, weighted_f1=%.4f",
        metrics["accuracy"],
        metrics["f1_weighted"],
    )
    return TrainingReport(
        model_path=model_output,
        metrics_path=metrics_output,
        train_rows=len(train_data),
        validation_rows=len(evaluation_data),
        feature_count=len(feature_columns),
        numeric_feature_count=len(numeric_columns),
        categorical_feature_count=len(categorical_columns),
        classes=fitted_classes,
        evaluation_dataset=evaluation_name,
        accuracy=metrics["accuracy"],
        f1_weighted=metrics["f1_weighted"],
    )


def main() -> None:
    """Run model training from the command line."""

    parser = argparse.ArgumentParser(
        description="Train and save a classification pipeline."
    )
    parser.add_argument("train_data", type=Path, help="Processed training data")
    parser.add_argument(
        "--target-column",
        required=True,
        help="Prediction target column",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        help="Processed validation data; defaults to validation.* beside train.*",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )
    parser.add_argument(
        "--model-type",
        choices=("logistic_regression", "random_forest"),
        default="logistic_regression",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable balanced class weighting",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing model and metrics files",
    )
    args = parser.parse_args()

    config = TrainConfig(
        train_path=args.train_data,
        validation_path=args.validation_data,
        target_column=args.target_column,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        model_type=args.model_type,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_estimators=args.n_estimators,
        class_weight=None if args.no_class_weight else "balanced",
        overwrite=args.overwrite,
    )
    report = train_model(config)
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
        LOGGER.exception("Model training failed.")
        raise
