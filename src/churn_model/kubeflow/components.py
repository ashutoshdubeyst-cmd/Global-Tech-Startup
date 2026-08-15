"""Kubeflow v2 container-component definitions for the training workflow.

The KFP SDK is intentionally imported only from this optional module. Runtime
containers use the project's normal image and invoke ``kubeflow.runtime``;
they do not install or import KFP in the pod.
"""

from dataclasses import dataclass
from typing import Any

try:
    from kfp import dsl
except ModuleNotFoundError as error:  # pragma: no cover - environment specific
    raise ModuleNotFoundError(
        "Kubeflow compilation requires the optional KFP SDK. Install "
        "requirements-kubeflow.txt before importing churn_model.kubeflow modules."
    ) from error


RUNTIME_MODULE = "src.churn_model.kubeflow.runtime"


def validate_component_image(component_image: str) -> str:
    """Validate and normalize the image embedded in compiled component specs."""

    if not isinstance(component_image, str) or not component_image.strip():
        raise ValueError("component_image cannot be empty.")
    image = component_image.strip()
    if any(character.isspace() for character in image):
        raise ValueError("component_image cannot contain whitespace.")
    return image


@dataclass(frozen=True)
class TrainingComponents:
    """Component factories used by the startup training pipeline."""

    validate_dataset: Any
    clean_dataset: Any
    build_feature_splits: Any
    train_classifier: Any
    evaluate_classifier: Any
    quality_gate: Any
    package_manifest: Any


def create_components(component_image: str) -> TrainingComponents:
    """Create container components bound to one immutable project image."""

    image = validate_component_image(component_image)

    @dsl.container_component
    def validate_dataset(
        input_dataset: dsl.Input[dsl.Dataset],
        validation_report: dsl.Output[dsl.Artifact],
        row_count: dsl.OutputPath(int),
        column_count: dsl.OutputPath(int),
        input_format: str,
        required_columns: list[str],
        target_column: str,
        allowed_target_values: list[str],
        max_missing_fraction: float,
        fail_on_duplicate_rows: bool,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "validate"],
            args=[
                "--input-path",
                input_dataset.path,
                "--input-format",
                input_format,
                "--report-output-path",
                validation_report.path,
                "--required-columns-json",
                required_columns,
                "--target-column",
                target_column,
                "--allowed-target-values-json",
                allowed_target_values,
                "--max-missing-fraction",
                max_missing_fraction,
                "--fail-on-duplicate-rows",
                fail_on_duplicate_rows,
                "--row-count-output-path",
                row_count,
                "--column-count-output-path",
                column_count,
            ],
        )

    @dsl.container_component
    def clean_dataset(
        input_dataset: dsl.Input[dsl.Dataset],
        cleaned_dataset: dsl.Output[dsl.Dataset],
        cleaning_report: dsl.Output[dsl.Artifact],
        input_rows: dsl.OutputPath(int),
        output_rows: dsl.OutputPath(int),
        input_format: str,
        output_format: str,
        numeric_columns: list[str],
        date_columns: list[str],
        missing_markers: list[str],
        drop_duplicate_rows: bool,
        drop_empty_rows: bool,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "clean"],
            args=[
                "--input-path",
                input_dataset.path,
                "--input-format",
                input_format,
                "--output-path",
                cleaned_dataset.path,
                "--output-format",
                output_format,
                "--report-output-path",
                cleaning_report.path,
                "--numeric-columns-json",
                numeric_columns,
                "--date-columns-json",
                date_columns,
                "--missing-markers-json",
                missing_markers,
                "--drop-duplicate-rows",
                drop_duplicate_rows,
                "--drop-empty-rows",
                drop_empty_rows,
                "--input-rows-output-path",
                input_rows,
                "--output-rows-output-path",
                output_rows,
            ],
        )

    @dsl.container_component
    def build_feature_splits(
        cleaned_dataset: dsl.Input[dsl.Dataset],
        train_dataset: dsl.Output[dsl.Dataset],
        validation_dataset: dsl.Output[dsl.Dataset],
        test_dataset: dsl.Output[dsl.Dataset],
        feature_metadata: dsl.Output[dsl.Artifact],
        train_rows: dsl.OutputPath(int),
        validation_rows: dsl.OutputPath(int),
        test_rows: dsl.OutputPath(int),
        input_format: str,
        output_format: str,
        target_column: str,
        drop_columns: list[str],
        date_columns: list[str],
        keep_date_columns: bool,
        age_from_year_columns: list[str],
        reference_year: int,
        log_columns: list[str],
        ratio_features: list[str],
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
        random_state: int,
        stratify: bool,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "build-features"],
            args=[
                "--input-path",
                cleaned_dataset.path,
                "--input-format",
                input_format,
                "--train-output-path",
                train_dataset.path,
                "--validation-output-path",
                validation_dataset.path,
                "--test-output-path",
                test_dataset.path,
                "--metadata-output-path",
                feature_metadata.path,
                "--target-column",
                target_column,
                "--output-format",
                output_format,
                "--drop-columns-json",
                drop_columns,
                "--date-columns-json",
                date_columns,
                "--keep-date-columns",
                keep_date_columns,
                "--age-from-year-columns-json",
                age_from_year_columns,
                "--reference-year",
                reference_year,
                "--log-columns-json",
                log_columns,
                "--ratio-features-json",
                ratio_features,
                "--train-fraction",
                train_fraction,
                "--validation-fraction",
                validation_fraction,
                "--test-fraction",
                test_fraction,
                "--random-state",
                random_state,
                "--stratify",
                stratify,
                "--train-rows-output-path",
                train_rows,
                "--validation-rows-output-path",
                validation_rows,
                "--test-rows-output-path",
                test_rows,
            ],
        )

    @dsl.container_component
    def train_classifier(
        train_dataset: dsl.Input[dsl.Dataset],
        validation_dataset: dsl.Input[dsl.Dataset],
        candidate_model: dsl.Output[dsl.Model],
        training_metrics: dsl.Output[dsl.Metrics],
        validation_accuracy: dsl.OutputPath(float),
        validation_f1_weighted: dsl.OutputPath(float),
        data_format: str,
        target_column: str,
        model_type: str,
        random_state: int,
        max_iter: int,
        n_estimators: int,
        class_weight: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "train"],
            args=[
                "--train-path",
                train_dataset.path,
                "--validation-path",
                validation_dataset.path,
                "--data-format",
                data_format,
                "--model-output-path",
                candidate_model.path,
                "--metrics-output-path",
                training_metrics.path,
                "--target-column",
                target_column,
                "--model-type",
                model_type,
                "--random-state",
                random_state,
                "--max-iter",
                max_iter,
                "--n-estimators",
                n_estimators,
                "--class-weight",
                class_weight,
                "--accuracy-output-path",
                validation_accuracy,
                "--f1-weighted-output-path",
                validation_f1_weighted,
            ],
        )

    @dsl.container_component
    def evaluate_classifier(
        test_dataset: dsl.Input[dsl.Dataset],
        candidate_model: dsl.Input[dsl.Model],
        evaluation_metrics: dsl.Output[dsl.Metrics],
        test_accuracy: dsl.OutputPath(float),
        test_balanced_accuracy: dsl.OutputPath(float),
        test_f1_weighted: dsl.OutputPath(float),
        data_format: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "evaluate"],
            args=[
                "--data-path",
                test_dataset.path,
                "--data-format",
                data_format,
                "--model-path",
                candidate_model.path,
                "--report-output-path",
                evaluation_metrics.path,
                "--accuracy-output-path",
                test_accuracy,
                "--balanced-accuracy-output-path",
                test_balanced_accuracy,
                "--f1-weighted-output-path",
                test_f1_weighted,
            ],
        )

    @dsl.container_component
    def quality_gate(
        approval_decision: dsl.Output[dsl.Artifact],
        passed: dsl.OutputPath(bool),
        metric_value: float,
        minimum_value: float,
        metric_name: str,
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "quality-gate"],
            args=[
                "--metric-value",
                metric_value,
                "--minimum-value",
                minimum_value,
                "--metric-name",
                metric_name,
                "--approval-output-path",
                approval_decision.path,
                "--passed-output-path",
                passed,
            ],
        )

    @dsl.container_component
    def package_manifest(
        candidate_model: dsl.Input[dsl.Model],
        training_metrics: dsl.Input[dsl.Metrics],
        evaluation_metrics: dsl.Input[dsl.Metrics],
        manifest: dsl.Output[dsl.Artifact],
        model_filename: str,
        drop_columns: list[str],
        date_columns: list[str],
        keep_date_columns: bool,
        age_from_year_columns: list[str],
        reference_year: int,
        log_columns: list[str],
        ratio_features: list[str],
    ):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", RUNTIME_MODULE, "package"],
            args=[
                "--model-path",
                candidate_model.path,
                "--training-metrics-path",
                training_metrics.path,
                "--evaluation-metrics-path",
                evaluation_metrics.path,
                "--manifest-output-path",
                manifest.path,
                "--model-filename",
                model_filename,
                "--drop-columns-json",
                drop_columns,
                "--date-columns-json",
                date_columns,
                "--keep-date-columns",
                keep_date_columns,
                "--age-from-year-columns-json",
                age_from_year_columns,
                "--reference-year",
                reference_year,
                "--log-columns-json",
                log_columns,
                "--ratio-features-json",
                ratio_features,
            ],
        )

    return TrainingComponents(
        validate_dataset=validate_dataset,
        clean_dataset=clean_dataset,
        build_feature_splits=build_feature_splits,
        train_classifier=train_classifier,
        evaluate_classifier=evaluate_classifier,
        quality_gate=quality_gate,
        package_manifest=package_manifest,
    )
