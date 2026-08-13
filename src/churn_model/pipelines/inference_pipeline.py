"""Run feature engineering and prediction using saved training artifacts.

For cleaned inputs, this pipeline reads ``training_pipeline_manifest.json``
and repeats the exact deterministic feature rules used during training. The
fitted imputation, encoding, and scaling steps are never refitted; they are
loaded from the trusted pickle artifact and applied by the saved sklearn
pipeline.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from ..data.ingestion import load_dataset
    from ..features.build_features import FeatureBuildConfig, build_features
    from ..models.predict import (
        DEFAULT_MODEL_PATH,
        DEFAULT_PREDICTIONS_PATH,
        PredictionConfig,
        PredictionReport,
        predict_data,
    )
    from ..models.train import load_model_artifact
    from .training_pipeline import (
        DEFAULT_MANIFEST_PATH,
        load_training_manifest,
    )
except ImportError:
    # Support VS Code's "Run Python File" command.
    import sys

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from src.churn_model.data.ingestion import load_dataset
    from src.churn_model.features.build_features import (
        FeatureBuildConfig,
        build_features,
    )
    from src.churn_model.models.predict import (
        DEFAULT_MODEL_PATH,
        DEFAULT_PREDICTIONS_PATH,
        PredictionConfig,
        PredictionReport,
        predict_data,
    )
    from src.churn_model.models.train import load_model_artifact
    from src.churn_model.pipelines.training_pipeline import (
        DEFAULT_MANIFEST_PATH,
        load_training_manifest,
    )


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INFERENCE_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "inference_features.csv"
)


@dataclass(frozen=True)
class InferencePipelineConfig:
    """Configuration for feature preparation and batch prediction."""

    data_path: Path
    model_path: Path = DEFAULT_MODEL_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    features_output: Path = DEFAULT_INFERENCE_FEATURES_PATH
    predictions_output: Path = DEFAULT_PREDICTIONS_PATH
    input_stage: str = "auto"
    prediction_column: str | None = None
    include_input: bool = True
    include_probabilities: bool = True
    preserve_original_columns: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.input_stage not in {"auto", "cleaned", "processed"}:
            raise ValueError(
                "input_stage must be 'auto', 'cleaned', or 'processed'."
            )
        if self.model_path.suffix.lower() not in {".pkl", ".joblib"}:
            raise ValueError("model_path must end with '.pkl' or '.joblib'.")


@dataclass(frozen=True)
class InferencePipelineReport:
    """Summary of a completed inference run."""

    resolved_input_stage: str
    data_path: Path
    model_path: Path
    features_path: Path | None
    prediction_report: PredictionReport
    engineered_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """Return a readable inference summary."""

        lines = [
            "Inference pipeline completed.",
            f"Input stage: {self.resolved_input_stage}",
            f"Input data: {self.data_path}",
            f"Predicted rows: {self.prediction_report.predicted_rows}",
            f"Prediction column: {self.prediction_report.prediction_column}",
            f"Model artifact: {self.model_path}",
            f"Predictions: {self.prediction_report.output_path}",
        ]
        if self.features_path is not None:
            lines.append(f"Inference features: {self.features_path}")
        if self.engineered_features:
            lines.append(
                f"Engineered features: {list(self.engineered_features)}"
            )
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _validate_paths(
    config: InferencePipelineConfig,
    resolved_stage: str,
) -> None:
    """Prevent inputs, artifacts, and outputs from overwriting one another."""

    data_path = config.data_path.expanduser().resolve()
    model_path = config.model_path.expanduser().resolve()
    manifest_path = config.manifest_path.expanduser().resolve()
    predictions_path = config.predictions_output.expanduser().resolve()
    features_path = config.features_output.expanduser().resolve()

    protected_inputs = {data_path, model_path}
    if resolved_stage == "cleaned":
        protected_inputs.add(manifest_path)
    if predictions_path in protected_inputs:
        raise ValueError(
            "Prediction output cannot overwrite input data, the model, or the "
            "training manifest."
        )
    if resolved_stage == "cleaned":
        if features_path in protected_inputs or features_path == predictions_path:
            raise ValueError(
                "Inference-feature output must differ from all inputs and the "
                "prediction output."
            )

    outputs = [predictions_path]
    if resolved_stage == "cleaned":
        outputs.append(features_path)
    if not config.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Inference outputs already exist: "
                f"{', '.join(str(path) for path in existing)}. "
                "Use --overwrite to replace them."
            )


def _resolve_input_stage(
    requested_stage: str,
    data: pd.DataFrame,
    expected_features: tuple[str, ...],
) -> str:
    """Resolve automatic input-stage detection using the model schema."""

    if requested_stage != "auto":
        return requested_stage
    if set(expected_features).issubset(data.columns):
        LOGGER.info(
            "Input-stage auto-detection selected processed data because every "
            "model feature is already present"
        )
        return "processed"
    LOGGER.info(
        "Input-stage auto-detection selected cleaned data because engineered "
        "model features are missing"
    )
    return "cleaned"


def _validate_manifest_against_model(
    manifest: dict[str, object],
    artifact: dict[str, object],
    model_path: Path,
) -> None:
    """Ensure feature rules belong to the selected model artifact."""

    manifest_features = tuple(manifest["expected_feature_columns"])
    artifact_features = tuple(artifact["feature_columns"])
    if manifest_features != artifact_features:
        raise ValueError(
            "Training manifest and model artifact expect different feature "
            "schemas. Use artifacts from the same training run."
        )
    if manifest["target_column"] != artifact["target_column"]:
        raise ValueError(
            "Training manifest and model artifact use different targets."
        )

    recorded_model = Path(str(manifest["model_path"])).expanduser().resolve()
    if recorded_model != model_path:
        LOGGER.warning(
            "The selected model path differs from the path recorded in the "
            "manifest; schema compatibility was verified successfully: %s",
            model_path,
        )


def _feature_config_from_manifest(
    data_path: Path,
    manifest: dict[str, object],
) -> FeatureBuildConfig:
    """Reconstruct deterministic feature rules saved during training."""

    values = manifest["feature_build_config"]
    if not isinstance(values, dict):
        raise ValueError("Manifest feature_build_config must be an object.")
    required = {
        "drop_columns",
        "date_columns",
        "keep_date_columns",
        "age_from_year_columns",
        "reference_year",
        "log_columns",
        "ratio_features",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(
            f"Manifest feature-building configuration is incomplete: {missing}"
        )

    return FeatureBuildConfig(
        source_path=data_path,
        target_column=None,
        drop_columns=tuple(values["drop_columns"]),
        date_columns=tuple(values["date_columns"]),
        keep_date_columns=bool(values["keep_date_columns"]),
        age_from_year_columns=tuple(values["age_from_year_columns"]),
        reference_year=values["reference_year"],
        log_columns=tuple(values["log_columns"]),
        ratio_features=tuple(values["ratio_features"]),
    )


def _save_features(data: pd.DataFrame, path: Path) -> None:
    """Save inference features as CSV or Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        data.to_csv(path, index=False)
    elif suffix == ".parquet":
        data.to_parquet(path, index=False)
    else:
        raise ValueError(
            "Inference-feature output must end with '.csv' or '.parquet'."
        )


def _build_inference_features(
    data: pd.DataFrame,
    data_path: Path,
    manifest: dict[str, object],
    config: InferencePipelineConfig,
) -> tuple[Path, tuple[str, ...], tuple[str, ...]]:
    """Apply saved feature rules and persist the model-ready table."""

    feature_config = _feature_config_from_manifest(data_path, manifest)
    feature_data, engineered_features, warnings, _ = build_features(
        data,
        feature_config,
    )
    if len(feature_data) != len(data):
        raise RuntimeError(
            "Inference feature building changed the number of rows."
        )

    if config.preserve_original_columns:
        original = data.reset_index(drop=True)
        for column in original.columns:
            if column not in feature_data.columns:
                feature_data[column] = original[column]

    expected_features = tuple(manifest["expected_feature_columns"])
    missing_features = sorted(set(expected_features) - set(feature_data.columns))
    if missing_features:
        raise ValueError(
            "Feature engineering did not produce model features: "
            f"{missing_features}"
        )

    features_path = config.features_output.expanduser().resolve()
    _save_features(feature_data, features_path)
    LOGGER.info(
        "Saved inference features: rows=%s, columns=%s, path=%s",
        len(feature_data),
        len(feature_data.columns),
        features_path,
    )
    return features_path, engineered_features, warnings


def run_inference_pipeline(
    config: InferencePipelineConfig,
) -> InferencePipelineReport:
    """Prepare input features and generate predictions with a saved model."""

    data_path = config.data_path.expanduser().resolve()
    model_path = config.model_path.expanduser().resolve()
    LOGGER.info(
        "Starting inference pipeline: data=%s, model=%s, requested_stage=%s",
        data_path,
        model_path,
        config.input_stage,
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"Inference dataset was not found: {data_path}")

    stage = "artifact loading"
    try:
        LOGGER.info("Inference pipeline stage started: %s", stage)
        artifact = load_model_artifact(model_path)
        data = load_dataset(data_path)
        if data.empty:
            raise ValueError("Inference dataset contains no rows.")
        resolved_stage = _resolve_input_stage(
            config.input_stage,
            data,
            tuple(artifact["feature_columns"]),
        )
        _validate_paths(config, resolved_stage)
        LOGGER.info(
            "Inference pipeline stage completed: %s; resolved_stage=%s",
            stage,
            resolved_stage,
        )

        engineered_features: tuple[str, ...] = ()
        warnings: tuple[str, ...] = ()
        features_path: Path | None = None
        prediction_input = data_path
        if resolved_stage == "cleaned":
            stage = "feature building"
            LOGGER.info("Inference pipeline stage started: %s", stage)
            manifest = load_training_manifest(config.manifest_path)
            _validate_manifest_against_model(manifest, artifact, model_path)
            features_path, engineered_features, warnings = (
                _build_inference_features(
                    data,
                    data_path,
                    manifest,
                    config,
                )
            )
            prediction_input = features_path
            for warning in warnings:
                LOGGER.warning("Inference feature warning: %s", warning)
            LOGGER.info("Inference pipeline stage completed: %s", stage)

        stage = "prediction"
        LOGGER.info("Inference pipeline stage started: %s", stage)
        prediction_report = predict_data(
            PredictionConfig(
                data_path=prediction_input,
                model_path=model_path,
                output_path=config.predictions_output,
                prediction_column=config.prediction_column,
                include_input=config.include_input,
                include_probabilities=config.include_probabilities,
                overwrite=config.overwrite,
            )
        )
        LOGGER.info("Inference pipeline stage completed: %s", stage)
    except Exception:
        LOGGER.exception("Inference pipeline failed during stage: %s", stage)
        raise

    report = InferencePipelineReport(
        resolved_input_stage=resolved_stage,
        data_path=data_path,
        model_path=model_path,
        features_path=features_path,
        prediction_report=prediction_report,
        engineered_features=engineered_features,
        warnings=warnings,
    )
    LOGGER.info(
        "Inference pipeline completed: rows=%s, predictions=%s",
        prediction_report.predicted_rows,
        prediction_report.output_path,
    )
    return report


def main() -> None:
    """Run the inference pipeline from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Apply saved feature rules and generate predictions with a "
            "trusted pickle model."
        )
    )
    parser.add_argument("data", type=Path, help="Cleaned or processed data")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Training manifest required for cleaned input",
    )
    parser.add_argument(
        "--input-stage",
        choices=("auto", "cleaned", "processed"),
        default="auto",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=DEFAULT_INFERENCE_FEATURES_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument("--prediction-column")
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--no-probabilities", action="store_true")
    parser.add_argument(
        "--do-not-preserve-original-columns",
        action="store_true",
        help="Exclude cleaned input columns removed during feature building",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = run_inference_pipeline(
        InferencePipelineConfig(
            data_path=args.data,
            model_path=args.model,
            manifest_path=args.manifest,
            features_output=args.features_output,
            predictions_output=args.output,
            input_stage=args.input_stage,
            prediction_column=args.prediction_column,
            include_input=not args.predictions_only,
            include_probabilities=not args.no_probabilities,
            preserve_original_columns=(
                not args.do_not_preserve_original_columns
            ),
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
