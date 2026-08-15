"""Kubeflow v2 training pipeline for the startup acquisition model."""

from typing import NamedTuple

try:
    from kfp import dsl
except ModuleNotFoundError as error:  # pragma: no cover - environment specific
    raise ModuleNotFoundError(
        "Kubeflow compilation requires the optional KFP SDK. Install "
        "requirements-kubeflow.txt before importing churn_model.kubeflow modules."
    ) from error

from .components import create_components


PIPELINE_NAME = "startup-acquisition-training"
LOGICAL_MODEL_FILENAME = "startup_classifier.pkl"


StartupPipelineOutputs = NamedTuple(
    "StartupPipelineOutputs",
    [
        ("candidate_model", dsl.Model),
        ("manifest", dsl.Artifact),
        ("evaluation_metrics", dsl.Metrics),
        ("approval_decision", dsl.Artifact),
    ],
)


def create_startup_training_pipeline(component_image: str):
    """Create a pipeline definition bound to a compile-time container image."""

    components = create_components(component_image)

    @dsl.pipeline(
        name=PIPELINE_NAME,
        description=(
            "Validate, clean, feature engineer, train, evaluate, and quality "
            "gate the startup acquisition classifier."
        ),
    )
    def startup_training_pipeline(
        source_dataset_uri: str,
        source_format: str = "csv",
        target_column: str = "acquisition_status",
        required_columns: list[str] = [
            "company_id",
            "domain",
            "founding_year",
            "country",
            "city",
            "funding_stage",
            "total_funding_usd_millions",
            "valuation_usd_millions",
            "revenue_arr_millions",
            "monthly_burn_rate_millions",
            "runway_months_2024",
            "peak_headcount_2023",
            "layoffs_2024_2025",
            "current_headcount_2026",
            "investor_tier",
            "ai_adoption_level",
        ],
        allowed_target_values: list[str] = ["Acquired", "Independent"],
        max_missing_fraction: float = 1.0,
        fail_on_duplicate_rows: bool = False,
        numeric_columns: list[str] = [
            "founding_year",
            "total_funding_usd_millions",
            "valuation_usd_millions",
            "revenue_arr_millions",
            "monthly_burn_rate_millions",
            "runway_months_2024",
            "peak_headcount_2023",
            "layoffs_2024_2025",
            "current_headcount_2026",
        ],
        cleaning_date_columns: list[str] = [],
        missing_markers: list[str] = ["", "n/a", "null", "none", "nan"],
        drop_duplicate_rows: bool = True,
        drop_empty_rows: bool = True,
        drop_columns: list[str] = ["company_id"],
        feature_date_columns: list[str] = [],
        keep_date_columns: bool = False,
        age_from_year_columns: list[str] = ["founding_year"],
        reference_year: int = 2026,
        log_columns: list[str] = [
            "total_funding_usd_millions",
            "valuation_usd_millions",
            "revenue_arr_millions",
        ],
        ratio_features: list[str] = [
            "funding_per_employee="
            "total_funding_usd_millions/current_headcount_2026",
            "valuation_to_revenue="
            "valuation_usd_millions/revenue_arr_millions",
        ],
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        random_state: int = 42,
        stratify: bool = True,
        output_format: str = "parquet",
        model_type: str = "logistic_regression",
        max_iter: int = 2000,
        n_estimators: int = 300,
        class_weight: str = "balanced",
        approval_metric: str = "f1_weighted",
        minimum_approval_metric: float = 0.70,
    ) -> StartupPipelineOutputs:
        source = dsl.importer(
            artifact_uri=source_dataset_uri,
            artifact_class=dsl.Dataset,
            # KFP 2.x requires a compile-time bool here; pipeline parameter
            # channels are supported for artifact_uri but not for reimport.
            reimport=True,
        )
        source.set_display_name("Import source startup dataset")

        raw_validation = components.validate_dataset(
            input_dataset=source.output,
            input_format=source_format,
            required_columns=required_columns,
            target_column=target_column,
            allowed_target_values=allowed_target_values,
            max_missing_fraction=max_missing_fraction,
            fail_on_duplicate_rows=fail_on_duplicate_rows,
        )
        raw_validation.set_display_name("Validate raw startup data")

        cleaning = components.clean_dataset(
            input_dataset=source.output,
            input_format=source_format,
            output_format=output_format,
            numeric_columns=numeric_columns,
            date_columns=cleaning_date_columns,
            missing_markers=missing_markers,
            drop_duplicate_rows=drop_duplicate_rows,
            drop_empty_rows=drop_empty_rows,
        ).after(raw_validation)
        cleaning.set_display_name("Clean startup data")

        cleaned_validation = components.validate_dataset(
            input_dataset=cleaning.outputs["cleaned_dataset"],
            input_format=output_format,
            required_columns=required_columns,
            target_column=target_column,
            allowed_target_values=allowed_target_values,
            max_missing_fraction=max_missing_fraction,
            fail_on_duplicate_rows=True,
        )
        cleaned_validation.set_display_name("Validate cleaned startup data")

        features = components.build_feature_splits(
            cleaned_dataset=cleaning.outputs["cleaned_dataset"],
            input_format=output_format,
            output_format=output_format,
            target_column=target_column,
            drop_columns=drop_columns,
            date_columns=feature_date_columns,
            keep_date_columns=keep_date_columns,
            age_from_year_columns=age_from_year_columns,
            reference_year=reference_year,
            log_columns=log_columns,
            ratio_features=ratio_features,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_state=random_state,
            stratify=stratify,
        ).after(cleaned_validation)
        features.set_display_name("Build and split model features")

        training = components.train_classifier(
            train_dataset=features.outputs["train_dataset"],
            validation_dataset=features.outputs["validation_dataset"],
            data_format=output_format,
            target_column=target_column,
            model_type=model_type,
            random_state=random_state,
            max_iter=max_iter,
            n_estimators=n_estimators,
            class_weight=class_weight,
        )
        training.set_display_name("Train startup classifier")

        evaluation = components.evaluate_classifier(
            test_dataset=features.outputs["test_dataset"],
            candidate_model=training.outputs["candidate_model"],
            data_format=output_format,
        )
        evaluation.set_display_name("Evaluate held-out test data")

        with dsl.If(approval_metric == "accuracy", name="select-accuracy"):
            accuracy_gate = components.quality_gate(
                metric_value=evaluation.outputs["test_accuracy"],
                minimum_value=minimum_approval_metric,
                metric_name=approval_metric,
            )
        with dsl.Elif(
            approval_metric == "balanced_accuracy",
            name="select-balanced-accuracy",
        ):
            balanced_accuracy_gate = components.quality_gate(
                metric_value=evaluation.outputs["test_balanced_accuracy"],
                minimum_value=minimum_approval_metric,
                metric_name=approval_metric,
            )
        with dsl.Else(name="select-weighted-f1"):
            f1_gate = components.quality_gate(
                metric_value=evaluation.outputs["test_f1_weighted"],
                minimum_value=minimum_approval_metric,
                metric_name=approval_metric,
            )

        gate_passed = dsl.OneOf(
            accuracy_gate.outputs["passed"],
            balanced_accuracy_gate.outputs["passed"],
            f1_gate.outputs["passed"],
        )
        approval_decision = dsl.OneOf(
            accuracy_gate.outputs["approval_decision"],
            balanced_accuracy_gate.outputs["approval_decision"],
            f1_gate.outputs["approval_decision"],
        )

        package_arguments = {
            "candidate_model": training.outputs["candidate_model"],
            "training_metrics": training.outputs["training_metrics"],
            "evaluation_metrics": evaluation.outputs["evaluation_metrics"],
            "model_filename": LOGICAL_MODEL_FILENAME,
            "drop_columns": drop_columns,
            "date_columns": feature_date_columns,
            "keep_date_columns": keep_date_columns,
            "age_from_year_columns": age_from_year_columns,
            "reference_year": reference_year,
            "log_columns": log_columns,
            "ratio_features": ratio_features,
        }
        with dsl.If(gate_passed == True, name="approved-model"):
            approved_manifest = components.package_manifest(**package_arguments)
            approved_manifest.set_display_name("Package approved model manifest")
        with dsl.Else(name="rejected-model"):
            rejected_manifest = components.package_manifest(**package_arguments)
            rejected_manifest.set_display_name("Package candidate model manifest")

        manifest = dsl.OneOf(
            approved_manifest.outputs["manifest"],
            rejected_manifest.outputs["manifest"],
        )
        return StartupPipelineOutputs(
            candidate_model=training.outputs["candidate_model"],
            manifest=manifest,
            evaluation_metrics=evaluation.outputs["evaluation_metrics"],
            approval_decision=approval_decision,
        )

    return startup_training_pipeline
