"""Runtime adapters used by the Kubeflow container components.

KFP artifact mounts do not preserve filename extensions.  The project's public
pipeline APIs deliberately use extensions to select their tabular reader and
writer, so these adapters stage extensionless artifacts in a temporary
directory and then publish the resulting files at the paths supplied by KFP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence

from ..data.cleaning import DEFAULT_MISSING_MARKERS, CleaningConfig, clean_data
from ..data.validation import validate_data
from ..features.build_features import FeatureBuildConfig, build_feature_datasets
from ..models.evaluate import EvaluationConfig, evaluate_model
from ..models.train import TrainConfig, load_model_artifact, train_model


FORMAT_EXTENSIONS = {"csv": ".csv", "parquet": ".parquet"}
APPROVAL_METRICS = ("accuracy", "balanced_accuracy", "f1_weighted")
MODEL_SUFFIXES = {".pkl", ".joblib"}
LOGGER = logging.getLogger(__name__)


def _format_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in FORMAT_EXTENSIONS:
        supported = ", ".join(sorted(FORMAT_EXTENSIONS))
        raise argparse.ArgumentTypeError(
            f"Unsupported data format {value!r}; choose one of: {supported}."
        )
    return normalized


def _boolean(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"Expected 'true' or 'false', received {value!r}."
    )


def _json_string_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        parsed: Any = value
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise argparse.ArgumentTypeError(
                "Expected a JSON array of strings."
            ) from error
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise argparse.ArgumentTypeError("Expected a JSON array of strings.")
    return parsed


def _optional_class_weight(value: str) -> str | None:
    return None if value.strip().lower() in {"", "none", "null"} else value


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_scalar(path: str | Path | None, value: Any) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    destination.write_text(text, encoding="utf-8")


def _stage_input(
    source_path: str | Path,
    suffix: str,
    temporary_directory: Path,
    stem: str,
) -> Path:
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input artifact was not found: {source}")
    if source.suffix.lower() == suffix:
        return source
    staged = temporary_directory / f"{stem}{suffix}"
    shutil.copyfile(source, staged)
    return staged


def _publish(source_path: str | Path, output_path: str | Path) -> None:
    source = Path(source_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _read_json_object(
    source_path: str | Path,
    temporary_directory: Path,
    stem: str,
) -> dict[str, Any]:
    staged = _stage_input(source_path, ".json", temporary_directory, stem)
    payload = json.loads(staged.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {source_path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_validate(args: argparse.Namespace) -> int:
    extension = FORMAT_EXTENSIONS[args.input_format]
    with TemporaryDirectory(prefix="kubeflow-validate-") as directory:
        staged_input = _stage_input(
            args.input_path,
            extension,
            Path(directory),
            "input",
        )
        report = validate_data(
            staged_input,
            required_columns=tuple(args.required_columns_json),
            target_column=args.target_column,
            allowed_target_values=tuple(args.allowed_target_values_json),
            max_missing_fraction=args.max_missing_fraction,
            fail_on_duplicate_rows=args.fail_on_duplicate_rows,
        )

    payload = _json_safe(report)
    payload.update({"status": "passed", "data_path": str(args.input_path)})
    _write_json(args.report_output_path, payload)
    _write_scalar(args.row_count_output_path, report.row_count)
    _write_scalar(args.column_count_output_path, report.column_count)
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    input_extension = FORMAT_EXTENSIONS[args.input_format]
    output_extension = FORMAT_EXTENSIONS[args.output_format]
    with TemporaryDirectory(prefix="kubeflow-clean-") as directory:
        temporary_directory = Path(directory)
        staged_input = _stage_input(
            args.input_path,
            input_extension,
            temporary_directory,
            "input",
        )
        staged_output = temporary_directory / f"cleaned{output_extension}"
        report = clean_data(
            CleaningConfig(
                source_path=staged_input,
                output_path=staged_output,
                numeric_columns=tuple(args.numeric_columns_json),
                date_columns=tuple(args.date_columns_json),
                missing_markers=tuple(args.missing_markers_json),
                drop_duplicate_rows=args.drop_duplicate_rows,
                drop_empty_rows=args.drop_empty_rows,
                overwrite=True,
            )
        )
        _publish(staged_output, args.output_path)

    payload = _json_safe(report)
    payload.update(
        {
            "source_path": str(args.input_path),
            "output_path": str(args.output_path),
        }
    )
    _write_json(args.report_output_path, payload)
    _write_scalar(args.input_rows_output_path, report.input_rows)
    _write_scalar(args.output_rows_output_path, report.output_rows)
    return 0


def _run_build_features(args: argparse.Namespace) -> int:
    input_extension = FORMAT_EXTENSIONS[args.input_format]
    with TemporaryDirectory(prefix="kubeflow-features-") as directory:
        temporary_directory = Path(directory)
        staged_input = _stage_input(
            args.input_path,
            input_extension,
            temporary_directory,
            "cleaned",
        )
        output_directory = temporary_directory / "outputs"
        report = build_feature_datasets(
            FeatureBuildConfig(
                source_path=staged_input,
                output_dir=output_directory,
                target_column=args.target_column,
                drop_columns=tuple(args.drop_columns_json),
                date_columns=tuple(args.date_columns_json),
                keep_date_columns=args.keep_date_columns,
                age_from_year_columns=tuple(args.age_from_year_columns_json),
                reference_year=args.reference_year,
                log_columns=tuple(args.log_columns_json),
                ratio_features=tuple(args.ratio_features_json),
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                random_state=args.random_state,
                stratify=args.stratify,
                output_format=args.output_format,
                overwrite=True,
            )
        )
        required_outputs = {
            "train": args.train_output_path,
            "validation": args.validation_output_path,
            "test": args.test_output_path,
            "metadata": args.metadata_output_path,
        }
        missing = sorted(set(required_outputs) - set(report.output_paths))
        if missing:
            raise ValueError(
                "Kubeflow feature component requires non-empty train, "
                f"validation, and test splits; missing outputs: {missing}."
            )
        for name, destination in required_outputs.items():
            _publish(report.output_paths[name], destination)

    _write_scalar(args.train_rows_output_path, report.split_rows["train"])
    _write_scalar(
        args.validation_rows_output_path,
        report.split_rows["validation"],
    )
    _write_scalar(args.test_rows_output_path, report.split_rows["test"])
    return 0


def _run_train(args: argparse.Namespace) -> int:
    extension = FORMAT_EXTENSIONS[args.data_format]
    with TemporaryDirectory(prefix="kubeflow-train-") as directory:
        temporary_directory = Path(directory)
        train_path = _stage_input(
            args.train_path,
            extension,
            temporary_directory,
            "train",
        )
        validation_path = _stage_input(
            args.validation_path,
            extension,
            temporary_directory,
            "validation",
        )
        model_path = temporary_directory / "startup_classifier.pkl"
        metrics_path = temporary_directory / "training_metrics.json"
        report = train_model(
            TrainConfig(
                train_path=train_path,
                validation_path=validation_path,
                target_column=args.target_column,
                model_output=model_path,
                metrics_output=metrics_path,
                model_type=args.model_type,
                random_state=args.random_state,
                max_iter=args.max_iter,
                n_estimators=args.n_estimators,
                class_weight=_optional_class_weight(args.class_weight),
                overwrite=True,
            )
        )
        _publish(model_path, args.model_output_path)
        metrics_payload = _read_json_object(
            metrics_path,
            temporary_directory,
            "published_training_metrics",
        )
        metrics_payload["model_path"] = "candidate_model"
        _write_json(args.metrics_output_path, metrics_payload)

    _write_scalar(args.accuracy_output_path, report.accuracy)
    _write_scalar(args.f1_weighted_output_path, report.f1_weighted)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    extension = FORMAT_EXTENSIONS[args.data_format]
    with TemporaryDirectory(prefix="kubeflow-evaluate-") as directory:
        temporary_directory = Path(directory)
        data_path = _stage_input(
            args.data_path,
            extension,
            temporary_directory,
            "test",
        )
        model_path = _stage_input(
            args.model_path,
            ".pkl",
            temporary_directory,
            "startup_classifier",
        )
        report_path = temporary_directory / "evaluation_metrics.json"
        report = evaluate_model(
            EvaluationConfig(
                data_path=data_path,
                model_path=model_path,
                report_path=report_path,
                overwrite=True,
            )
        )
        report_payload = _read_json_object(
            report_path,
            temporary_directory,
            "published_evaluation_metrics",
        )
        report_payload["data_path"] = "test_dataset"
        report_payload["model_path"] = "candidate_model"
        _write_json(args.report_output_path, report_payload)

    _write_scalar(args.accuracy_output_path, report.accuracy)
    _write_scalar(
        args.balanced_accuracy_output_path,
        report.balanced_accuracy,
    )
    _write_scalar(args.f1_weighted_output_path, report.f1_weighted)
    return 0


def _run_quality_gate(args: argparse.Namespace) -> int:
    if not math.isfinite(args.metric_value):
        raise ValueError("metric_value must be finite.")
    if not math.isfinite(args.minimum_value):
        raise ValueError("minimum_value must be finite.")
    if not 0.0 <= args.metric_value <= 1.0:
        raise ValueError("metric_value must be between 0.0 and 1.0.")
    if not 0.0 <= args.minimum_value <= 1.0:
        raise ValueError("minimum_value must be between 0.0 and 1.0.")
    passed = args.metric_value >= args.minimum_value
    _write_json(
        args.approval_output_path,
        {
            "metric_name": args.metric_name,
            "metric_value": args.metric_value,
            "minimum_value": args.minimum_value,
            "passed": passed,
            "decision": "approved" if passed else "rejected",
        },
    )
    _write_scalar(args.passed_output_path, passed)
    # A rejected candidate is a normal branch decision, not a task failure.
    return 0


def _run_package(args: argparse.Namespace) -> int:
    model_filename = args.model_filename.strip()
    if (
        not model_filename
        or "/" in model_filename
        or "\\" in model_filename
        or Path(model_filename).name != model_filename
    ):
        raise ValueError("model_filename must be a logical base filename.")
    if Path(model_filename).suffix.lower() not in MODEL_SUFFIXES:
        raise ValueError("model_filename must end with '.pkl' or '.joblib'.")

    with TemporaryDirectory(prefix="kubeflow-package-") as directory:
        temporary_directory = Path(directory)
        model_path = _stage_input(
            args.model_path,
            Path(model_filename).suffix or ".pkl",
            temporary_directory,
            "model",
        )
        training_metrics = _read_json_object(
            args.training_metrics_path,
            temporary_directory,
            "training_metrics",
        )
        evaluation_metrics = _read_json_object(
            args.evaluation_metrics_path,
            temporary_directory,
            "evaluation_metrics",
        )
        artifact = load_model_artifact(model_path)
        model_sha256 = _sha256(model_path)

    validation_values = training_metrics.get("metrics", {})
    test_values = evaluation_metrics.get("metrics", {})
    if not isinstance(validation_values, dict) or not isinstance(test_values, dict):
        raise ValueError("Training and evaluation metrics must contain objects.")
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": artifact["target_column"],
        "model_path": model_filename,
        "model_filename": model_filename,
        "model_sha256": model_sha256,
        "model_type": artifact.get("model_type"),
        "random_state": artifact.get("random_state"),
        "feature_build_config": {
            "drop_columns": args.drop_columns_json,
            "date_columns": args.date_columns_json,
            "keep_date_columns": args.keep_date_columns,
            "age_from_year_columns": args.age_from_year_columns_json,
            "reference_year": args.reference_year,
            "log_columns": args.log_columns_json,
            "ratio_features": args.ratio_features_json,
        },
        "expected_feature_columns": list(artifact["feature_columns"]),
        "numeric_columns": list(artifact["numeric_columns"]),
        "categorical_columns": list(artifact["categorical_columns"]),
        "classes": list(artifact["classes"]),
        "training_metrics_path": "training_metrics.json",
        "evaluation_metrics_path": "evaluation_metrics.json",
        "validation_metrics": validation_values,
        "test_metrics": test_values,
    }
    _write_json(args.manifest_output_path, manifest)
    return 0


def _add_path(
    parser: argparse.ArgumentParser,
    flag: str,
    *,
    required: bool = True,
) -> None:
    parser.add_argument(flag, required=required, type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by local tests and component containers."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    _add_path(validate, "--input-path")
    validate.add_argument("--input-format", required=True, type=_format_name)
    _add_path(validate, "--report-output-path")
    validate.add_argument(
        "--required-columns-json", type=_json_string_list, default=[]
    )
    validate.add_argument("--target-column")
    validate.add_argument(
        "--allowed-target-values-json", type=_json_string_list, default=[]
    )
    validate.add_argument("--max-missing-fraction", type=float, default=1.0)
    validate.add_argument(
        "--fail-on-duplicate-rows", type=_boolean, default=False
    )
    _add_path(validate, "--row-count-output-path", required=False)
    _add_path(validate, "--column-count-output-path", required=False)
    validate.set_defaults(handler=_run_validate)

    clean = commands.add_parser("clean")
    _add_path(clean, "--input-path")
    clean.add_argument("--input-format", required=True, type=_format_name)
    _add_path(clean, "--output-path")
    clean.add_argument("--output-format", required=True, type=_format_name)
    _add_path(clean, "--report-output-path")
    clean.add_argument(
        "--numeric-columns-json", type=_json_string_list, default=[]
    )
    clean.add_argument(
        "--date-columns-json", type=_json_string_list, default=[]
    )
    clean.add_argument(
        "--missing-markers-json",
        type=_json_string_list,
        default=list(DEFAULT_MISSING_MARKERS),
    )
    clean.add_argument("--drop-duplicate-rows", type=_boolean, default=True)
    clean.add_argument("--drop-empty-rows", type=_boolean, default=True)
    _add_path(clean, "--input-rows-output-path", required=False)
    _add_path(clean, "--output-rows-output-path", required=False)
    clean.set_defaults(handler=_run_clean)

    features = commands.add_parser("build-features")
    _add_path(features, "--input-path")
    features.add_argument("--input-format", required=True, type=_format_name)
    _add_path(features, "--train-output-path")
    _add_path(features, "--validation-output-path")
    _add_path(features, "--test-output-path")
    _add_path(features, "--metadata-output-path")
    features.add_argument("--target-column", required=True)
    features.add_argument("--output-format", required=True, type=_format_name)
    features.add_argument(
        "--drop-columns-json", type=_json_string_list, default=[]
    )
    features.add_argument(
        "--date-columns-json", type=_json_string_list, default=[]
    )
    features.add_argument("--keep-date-columns", type=_boolean, default=False)
    features.add_argument(
        "--age-from-year-columns-json", type=_json_string_list, default=[]
    )
    features.add_argument("--reference-year", type=int)
    features.add_argument(
        "--log-columns-json", type=_json_string_list, default=[]
    )
    features.add_argument(
        "--ratio-features-json", type=_json_string_list, default=[]
    )
    features.add_argument("--train-fraction", type=float, default=0.70)
    features.add_argument("--validation-fraction", type=float, default=0.15)
    features.add_argument("--test-fraction", type=float, default=0.15)
    features.add_argument("--random-state", type=int, default=42)
    features.add_argument("--stratify", type=_boolean, default=True)
    _add_path(features, "--train-rows-output-path", required=False)
    _add_path(features, "--validation-rows-output-path", required=False)
    _add_path(features, "--test-rows-output-path", required=False)
    features.set_defaults(handler=_run_build_features)

    train = commands.add_parser("train")
    _add_path(train, "--train-path")
    _add_path(train, "--validation-path")
    train.add_argument("--data-format", required=True, type=_format_name)
    _add_path(train, "--model-output-path")
    _add_path(train, "--metrics-output-path")
    train.add_argument("--target-column", required=True)
    train.add_argument(
        "--model-type",
        choices=("logistic_regression", "random_forest"),
        default="logistic_regression",
    )
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--max-iter", type=int, default=2000)
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced"
    )
    _add_path(train, "--accuracy-output-path", required=False)
    _add_path(train, "--f1-weighted-output-path", required=False)
    train.set_defaults(handler=_run_train)

    evaluate = commands.add_parser("evaluate")
    _add_path(evaluate, "--data-path")
    evaluate.add_argument("--data-format", required=True, type=_format_name)
    _add_path(evaluate, "--model-path")
    _add_path(evaluate, "--report-output-path")
    _add_path(evaluate, "--accuracy-output-path", required=False)
    _add_path(evaluate, "--balanced-accuracy-output-path", required=False)
    _add_path(evaluate, "--f1-weighted-output-path", required=False)
    evaluate.set_defaults(handler=_run_evaluate)

    gate = commands.add_parser("quality-gate")
    gate.add_argument("--metric-value", required=True, type=float)
    gate.add_argument("--minimum-value", required=True, type=float)
    gate.add_argument(
        "--metric-name", choices=APPROVAL_METRICS, default="f1_weighted"
    )
    _add_path(gate, "--approval-output-path")
    _add_path(gate, "--passed-output-path", required=False)
    gate.set_defaults(handler=_run_quality_gate)

    package = commands.add_parser("package")
    _add_path(package, "--model-path")
    _add_path(package, "--training-metrics-path")
    _add_path(package, "--evaluation-metrics-path")
    _add_path(package, "--manifest-output-path")
    package.add_argument("--model-filename", default="startup_classifier.pkl")
    package.add_argument(
        "--drop-columns-json", type=_json_string_list, default=[]
    )
    package.add_argument(
        "--date-columns-json", type=_json_string_list, default=[]
    )
    package.add_argument("--keep-date-columns", type=_boolean, default=False)
    package.add_argument(
        "--age-from-year-columns-json", type=_json_string_list, default=[]
    )
    package.add_argument("--reference-year", type=int)
    package.add_argument(
        "--log-columns-json", type=_json_string_list, default=[]
    )
    package.add_argument(
        "--ratio-features-json", type=_json_string_list, default=[]
    )
    package.set_defaults(handler=_run_package)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Kubeflow component command."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    LOGGER.info("Starting Kubeflow runtime stage: %s", args.command)
    result = handler(args)
    LOGGER.info("Kubeflow runtime stage completed: %s", args.command)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
