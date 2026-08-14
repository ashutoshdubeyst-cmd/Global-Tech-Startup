"""End-to-end test for training, evaluation, and cleaned-data inference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.churn_model.pipelines.inference_pipeline import (
    InferencePipelineConfig,
    run_inference_pipeline,
)
from src.churn_model.pipelines.training_pipeline import (
    TrainingPipelineConfig,
    run_training_pipeline,
)
from src.logger import setup_logging


def test_training_then_inference_end_to_end(
    tmp_path: Path,
    labelled_startups_path: Path,
    new_startups_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir, console=False)

    processed_dir = tmp_path / "processed"
    model_path = tmp_path / "models" / "startup_classifier.pkl"
    manifest_path = tmp_path / "models" / "training_pipeline_manifest.json"
    training_metrics = tmp_path / "reports" / "training_metrics.json"
    evaluation_metrics = tmp_path / "reports" / "evaluation_metrics.json"

    training_report = run_training_pipeline(
        TrainingPipelineConfig(
            source_path=labelled_startups_path,
            target_column="acquisition_status",
            processed_dir=processed_dir,
            model_path=model_path,
            training_metrics_path=training_metrics,
            evaluation_metrics_path=evaluation_metrics,
            manifest_path=manifest_path,
            drop_columns=("company_id",),
            age_from_year_columns=("founding_year",),
            reference_year=2026,
            log_columns=(
                "total_funding_usd_millions",
                "valuation_usd_millions",
                "revenue_arr_millions",
            ),
            ratio_features=(
                "funding_per_employee="
                "total_funding_usd_millions/current_headcount_2026",
            ),
            random_state=17,
        )
    )

    assert model_path.is_file()
    assert manifest_path.is_file()
    assert training_metrics.is_file()
    assert evaluation_metrics.is_file()
    assert (processed_dir / "train.csv").is_file()
    assert (processed_dir / "validation.csv").is_file()
    assert (processed_dir / "test.csv").is_file()
    assert sum(training_report.feature_report.split_rows.values()) == 24

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["target_column"] == "acquisition_status"
    assert manifest["model_path"] == str(model_path.resolve())
    assert "funding_per_employee" in manifest["expected_feature_columns"]

    predictions_path = tmp_path / "predictions" / "predictions.csv"
    inference_features = tmp_path / "processed" / "inference_features.csv"
    inference_report = run_inference_pipeline(
        InferencePipelineConfig(
            data_path=new_startups_path,
            model_path=model_path,
            manifest_path=manifest_path,
            features_output=inference_features,
            predictions_output=predictions_path,
            input_stage="auto",
        )
    )

    assert inference_report.resolved_input_stage == "cleaned"
    assert inference_features.is_file()
    assert predictions_path.is_file()

    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == 3
    assert "company_id" in predictions.columns
    assert "predicted_acquisition_status" in predictions.columns
    assert {
        "probability_acquired",
        "probability_independent",
    }.issubset(predictions.columns)

    log_text = (log_dir / "churn_model.log").read_text(encoding="utf-8")
    assert "Training pipeline completed" in log_text
    assert "Inference pipeline completed" in log_text
