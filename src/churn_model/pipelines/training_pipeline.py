"""Run feature building, model training, and test evaluation end to end.

The pipeline starts from a cleaned interim dataset. It creates reproducible
train/validation/test datasets, fits a preprocessing-and-classification
pipeline, saves it as a pickle artifact, evaluates it on the untouched test
split, and writes a manifest that the inference pipeline can reuse.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ..features.build_features import (
        DEFAULT_PROCESSED_DIR,
        FeatureBuildConfig,
        FeatureBuildReport,
        build_feature_datasets,
    )
    from ..models.evaluate import (
        DEFAULT_REPORT_PATH as DEFAULT_EVALUATION_METRICS_PATH,
        EvaluationConfig,
        EvaluationReport,
        evaluate_model,
    )
    from ..models.train import (
        DEFAULT_METRICS_PATH as DEFAULT_TRAINING_METRICS_PATH,
        DEFAULT_MODEL_PATH,
        TrainConfig,
        TrainingReport,
        load_model_artifact,
        save_json,
        train_model,
    )
except ImportError:
    # Support VS Code's "Run Python File" command.
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src.churn_model.features.build_features import (
        DEFAULT_PROCESSED_DIR,
        FeatureBuildConfig,
        FeatureBuildReport,
        build_feature_datasets,
    )
    from src.churn_model.models.evaluate import (
        DEFAULT_REPORT_PATH as DEFAULT_EVALUATION_METRICS_PATH,
        EvaluationConfig,
        EvaluationReport,
        evaluate_model,
    )
    from src.churn_model.models.train import (
        DEFAULT_METRICS_PATH as DEFAULT_TRAINING_METRICS_PATH,
        DEFAULT_MODEL_PATH,
        TrainConfig,
        TrainingReport,
        load_model_artifact,
        save_json,
        train_model,
    )


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "models" / "training_pipeline_manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class TrainingPipelineConfig:
    """Configuration for the end-to-end training pipeline."""

    source_path: Path
    target_column: str
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    model_path: Path = DEFAULT_MODEL_PATH
    training_metrics_path: Path = DEFAULT_TRAINING_METRICS_PATH
    evaluation_metrics_path: Path = DEFAULT_EVALUATION_METRICS_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH

    # Feature-building options shared later with inference.
    drop_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    keep_date_columns: bool = False
    age_from_year_columns: tuple[str, ...] = ()
    reference_year: int | None = None
    log_columns: tuple[str, ...] = ()
    ratio_features: tuple[str, ...] = ()

    # Dataset splitting options.
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42
    stratify: bool = True
    output_format: str = "csv"

    # Estimator options.
    model_type: str = "logistic_regression"
    max_iter: int = 2_000
    n_estimators: int = 300
    class_weight: str | None = "balanced"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.target_column.strip():
            raise ValueError("target_column cannot be empty.")
        if self.test_fraction <= 0:
            raise ValueError(
                "test_fraction must be greater than zero so final evaluation "
                "uses unseen data."
            )
        if self.validation_fraction <= 0:
            raise ValueError(
                "validation_fraction must be greater than zero so model "
                "selection does not rely on training metrics."
            )
        if self.model_path.suffix.lower() not in {".pkl", ".joblib"}:
            raise ValueError("model_path must end with '.pkl' or '.joblib'.")


@dataclass(frozen=True)
class TrainingPipelineReport:
    """Summary of every completed training-pipeline stage."""

    feature_report: FeatureBuildReport
    training_report: TrainingReport
    evaluation_report: EvaluationReport
    manifest_path: Path

    def summary(self) -> str:
        """Return a concise end-to-end summary."""

        return "\n".join(
            [
                "Training pipeline completed.",
                f"Processed rows: {self.feature_report.output_rows}",
                f"Split sizes: {self.feature_report.split_rows}",
                f"Validation accuracy: {self.training_report.accuracy:.4f}",
                f"Test accuracy: {self.evaluation_report.accuracy:.4f}",
                f"Test weighted F1: {self.evaluation_report.f1_weighted:.4f}",
                f"Model artifact: {self.training_report.model_path}",
                f"Pipeline manifest: {self.manifest_path}",
            ]
        )


def load_training_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a manifest created by this pipeline."""

    import json

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training-pipeline manifest was not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Training-pipeline manifest must contain an object.")

    required_keys = {
        "manifest_version",
        "target_column",
        "model_path",
        "feature_build_config",
        "expected_feature_columns",
    }
    missing_keys = sorted(required_keys - set(manifest))
    if missing_keys:
        raise ValueError(
            f"Training-pipeline manifest is missing fields: {missing_keys}"
        )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError(
            "Unsupported training-pipeline manifest version: "
            f"{manifest['manifest_version']}"
        )
    return manifest


def _expected_output_paths(config: TrainingPipelineConfig) -> tuple[Path, ...]:
    """Return every file this run is expected to create."""

    extension = ".csv" if config.output_format == "csv" else ".parquet"
    processed_outputs = [
        config.processed_dir / f"train{extension}",
        config.processed_dir / f"test{extension}",
        config.processed_dir / "feature_metadata.json",
    ]
    if config.validation_fraction > 0:
        processed_outputs.append(
            config.processed_dir / f"validation{extension}"
        )
    return tuple(
        path.expanduser().resolve()
        for path in (
            *processed_outputs,
            config.model_path,
            config.training_metrics_path,
            config.evaluation_metrics_path,
            config.manifest_path,
        )
    )


def _preflight_outputs(config: TrainingPipelineConfig) -> None:
    """Fail before stage one when outputs would be overwritten accidentally."""

    source = config.source_path.expanduser().resolve()
    outputs = _expected_output_paths(config)
    if len(outputs) != len(set(outputs)):
        duplicates = sorted(
            str(path)
            for path in set(outputs)
            if outputs.count(path) > 1
        )
        raise ValueError(
            f"Training output paths must be different: {duplicates}"
        )
    if source in outputs:
        raise ValueError(
            "A training output path resolves to the cleaned source dataset."
        )

    if config.overwrite:
        return
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Training-pipeline outputs already exist: "
            f"{', '.join(str(path) for path in existing)}. "
            "Use --overwrite to replace them."
        )


def _feature_config(config: TrainingPipelineConfig) -> FeatureBuildConfig:
    """Build the feature-stage configuration."""

    return FeatureBuildConfig(
        source_path=config.source_path,
        output_dir=config.processed_dir,
        target_column=config.target_column,
        drop_columns=config.drop_columns,
        date_columns=config.date_columns,
        keep_date_columns=config.keep_date_columns,
        age_from_year_columns=config.age_from_year_columns,
        reference_year=config.reference_year,
        log_columns=config.log_columns,
        ratio_features=config.ratio_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        random_state=config.random_state,
        stratify=config.stratify,
        output_format=config.output_format,
        overwrite=config.overwrite,
    )


def _manifest_payload(
    config: TrainingPipelineConfig,
    feature_report: FeatureBuildReport,
    training_report: TrainingReport,
    evaluation_report: EvaluationReport,
) -> dict[str, Any]:
    """Create reusable training and inference metadata."""

    artifact = load_model_artifact(training_report.model_path)
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(config.source_path.expanduser().resolve()),
        "target_column": config.target_column,
        "model_path": str(training_report.model_path),
        "model_type": config.model_type,
        "random_state": config.random_state,
        "feature_build_config": {
            "drop_columns": list(config.drop_columns),
            "date_columns": list(config.date_columns),
            "keep_date_columns": config.keep_date_columns,
            "age_from_year_columns": list(config.age_from_year_columns),
            "reference_year": config.reference_year,
            "log_columns": list(config.log_columns),
            "ratio_features": list(config.ratio_features),
        },
        "expected_feature_columns": list(artifact["feature_columns"]),
        "numeric_columns": list(artifact["numeric_columns"]),
        "categorical_columns": list(artifact["categorical_columns"]),
        "classes": list(artifact["classes"]),
        "processed_datasets": {
            name: str(path)
            for name, path in feature_report.output_paths.items()
        },
        "training_metrics_path": str(training_report.metrics_path),
        "evaluation_metrics_path": str(evaluation_report.report_path),
        "validation_metrics": {
            "accuracy": training_report.accuracy,
            "f1_weighted": training_report.f1_weighted,
        },
        "test_metrics": {
            "accuracy": evaluation_report.accuracy,
            "balanced_accuracy": evaluation_report.balanced_accuracy,
            "f1_weighted": evaluation_report.f1_weighted,
        },
    }


def run_training_pipeline(
    config: TrainingPipelineConfig,
) -> TrainingPipelineReport:
    """Run feature building, model training, and final evaluation."""

    source = config.source_path.expanduser().resolve()
    LOGGER.info(
        "Starting training pipeline: source=%s, target=%s",
        source,
        config.target_column,
    )
    if not source.is_file():
        raise FileNotFoundError(f"Cleaned training dataset was not found: {source}")

    # Constructing these validates split and estimator settings before writes.
    feature_config = _feature_config(config)
    train_config_template = TrainConfig(
        train_path=Path("train.placeholder"),
        target_column=config.target_column,
        model_output=config.model_path,
        metrics_output=config.training_metrics_path,
        model_type=config.model_type,
        random_state=config.random_state,
        max_iter=config.max_iter,
        n_estimators=config.n_estimators,
        class_weight=config.class_weight,
        overwrite=config.overwrite,
    )
    _preflight_outputs(config)

    stage = "feature building"
    try:
        LOGGER.info("Training pipeline stage started: %s", stage)
        feature_report = build_feature_datasets(feature_config)
        LOGGER.info("Training pipeline stage completed: %s", stage)

        train_path = feature_report.output_paths.get("train")
        test_path = feature_report.output_paths.get("test")
        validation_path = feature_report.output_paths.get("validation")
        if train_path is None or test_path is None:
            raise RuntimeError(
                "Feature building did not produce required train and test files."
            )

        stage = "model training"
        LOGGER.info("Training pipeline stage started: %s", stage)
        training_report = train_model(
            TrainConfig(
                train_path=train_path,
                validation_path=validation_path,
                target_column=train_config_template.target_column,
                model_output=train_config_template.model_output,
                metrics_output=train_config_template.metrics_output,
                model_type=train_config_template.model_type,
                random_state=train_config_template.random_state,
                max_iter=train_config_template.max_iter,
                n_estimators=train_config_template.n_estimators,
                class_weight=train_config_template.class_weight,
                overwrite=train_config_template.overwrite,
            )
        )
        LOGGER.info("Training pipeline stage completed: %s", stage)

        stage = "test evaluation"
        LOGGER.info("Training pipeline stage started: %s", stage)
        evaluation_report = evaluate_model(
            EvaluationConfig(
                data_path=test_path,
                model_path=training_report.model_path,
                report_path=config.evaluation_metrics_path,
                overwrite=config.overwrite,
            )
        )
        LOGGER.info("Training pipeline stage completed: %s", stage)

        stage = "manifest creation"
        LOGGER.info("Training pipeline stage started: %s", stage)
        manifest_path = config.manifest_path.expanduser().resolve()
        save_json(
            _manifest_payload(
                config,
                feature_report,
                training_report,
                evaluation_report,
            ),
            manifest_path,
        )
        LOGGER.info("Saved training-pipeline manifest to %s", manifest_path)
        LOGGER.info("Training pipeline stage completed: %s", stage)
    except Exception:
        LOGGER.exception("Training pipeline failed during stage: %s", stage)
        raise

    report = TrainingPipelineReport(
        feature_report=feature_report,
        training_report=training_report,
        evaluation_report=evaluation_report,
        manifest_path=manifest_path,
    )
    LOGGER.info(
        "Training pipeline completed: validation_accuracy=%.4f, "
        "test_accuracy=%.4f, model=%s",
        training_report.accuracy,
        evaluation_report.accuracy,
        training_report.model_path,
    )
    return report


def main() -> None:
    """Run the training pipeline from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Build features, train a classifier, and evaluate it on test data."
        )
    )
    parser.add_argument("source", type=Path, help="Cleaned interim dataset")
    parser.add_argument(
        "--target-column",
        required=True,
        help="Prediction target column",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--training-metrics-output",
        type=Path,
        default=DEFAULT_TRAINING_METRICS_PATH,
    )
    parser.add_argument(
        "--evaluation-metrics-output",
        type=Path,
        default=DEFAULT_EVALUATION_METRICS_PATH,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--drop-columns", nargs="*", default=())
    parser.add_argument("--date-columns", nargs="*", default=())
    parser.add_argument("--keep-date-columns", action="store_true")
    parser.add_argument("--age-from-year-columns", nargs="*", default=())
    parser.add_argument("--reference-year", type=int)
    parser.add_argument("--log-columns", nargs="*", default=())
    parser.add_argument(
        "--ratio-feature",
        action="append",
        default=[],
        help="Repeatable: new_feature=numerator/denominator",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument(
        "--output-format",
        choices=("csv", "parquet"),
        default="csv",
    )
    parser.add_argument(
        "--model-type",
        choices=("logistic_regression", "random_forest"),
        default="logistic_regression",
    )
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = run_training_pipeline(
        TrainingPipelineConfig(
            source_path=args.source,
            target_column=args.target_column,
            processed_dir=args.processed_dir,
            model_path=args.model_output,
            training_metrics_path=args.training_metrics_output,
            evaluation_metrics_path=args.evaluation_metrics_output,
            manifest_path=args.manifest_output,
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
            model_type=args.model_type,
            max_iter=args.max_iter,
            n_estimators=args.n_estimators,
            class_weight=None if args.no_class_weight else "balanced",
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
    main()
