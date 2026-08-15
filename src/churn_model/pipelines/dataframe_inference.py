"""In-memory inference for interactive applications and services.

This module applies feature rules from a trusted training manifest and uses the
saved sklearn pipeline without writing intermediate features or predictions to
shared filesystem paths.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from ..features.build_features import FeatureBuildConfig, build_features
from ..models.predict import predict_dataframe
from ..models.train import load_model_artifact
from .training_pipeline import load_training_manifest


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResources:
    """Validated model and manifest pair loaded from trusted local paths."""

    model_path: Path
    manifest_path: Path
    artifact: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DataFrameInferenceResult:
    """Result of an in-memory prediction request."""

    data: pd.DataFrame
    features: pd.DataFrame
    resolved_input_stage: str
    prediction_column: str
    probability_columns: tuple[str, ...]
    engineered_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _validated_name_sequence(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    """Validate an ordered sequence of unique, non-empty column names."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{field_name} must be a sequence of strings, not a string."
        )

    values = tuple(value)
    if not allow_empty and not values:
        raise ValueError(f"{field_name} cannot be empty.")

    invalid = [
        index
        for index, item in enumerate(values)
        if not isinstance(item, str) or not item.strip()
    ]
    if invalid:
        raise ValueError(
            f"{field_name} must contain non-empty strings; invalid indexes: "
            f"{invalid}."
        )
    if len(set(values)) != len(values):
        duplicates = sorted(
            {item for item in values if values.count(item) > 1}
        )
        raise ValueError(
            f"{field_name} contains duplicate values: {duplicates}."
        )
    return values


def _validated_classes(value: object, field_name: str) -> tuple[Any, ...]:
    """Validate an ordered, non-empty sequence of unique class labels."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{field_name} must be a sequence of class labels, not a string."
        )
    values = tuple(value)
    if not values:
        raise ValueError(f"{field_name} cannot be empty.")
    try:
        unique_count = len(set(values))
    except TypeError as error:
        raise ValueError(
            f"{field_name} must contain hashable class labels."
        ) from error
    if unique_count != len(values):
        raise ValueError(f"{field_name} contains duplicate class labels.")
    return values


def _validated_ratio_features(value: object) -> tuple[str, ...]:
    """Validate ratio-feature types, syntax, and output-name uniqueness."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            "feature_build_config.ratio_features must be a sequence of "
            "strings, not a string."
        )

    specifications = tuple(value)
    feature_names: list[str] = []
    for index, specification in enumerate(specifications):
        if not isinstance(specification, str):
            raise ValueError(
                "feature_build_config.ratio_features must contain strings; "
                f"invalid index: {index}."
            )
        if specification.count("=") != 1:
            raise ValueError(
                "Ratio features must use "
                "'new_feature=numerator/denominator'; "
                f"received: {specification!r}."
            )
        feature_name, expression = specification.split("=", maxsplit=1)
        if expression.count("/") != 1:
            raise ValueError(
                "Ratio features must use "
                "'new_feature=numerator/denominator'; "
                f"received: {specification!r}."
            )
        numerator, denominator = expression.split("/", maxsplit=1)
        parts = tuple(
            part.strip()
            for part in (feature_name, numerator, denominator)
        )
        if any(not part for part in parts):
            raise ValueError(
                "Ratio feature names and columns cannot be empty: "
                f"{specification!r}."
            )
        feature_names.append(parts[0])

    if len(set(feature_names)) != len(feature_names):
        duplicates = sorted(
            {name for name in feature_names if feature_names.count(name) > 1}
        )
        raise ValueError(
            "feature_build_config.ratio_features contains duplicate output "
            f"names: {duplicates}."
        )
    return specifications


def _validate_resource_pair(
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    model_path: Path,
) -> None:
    """Ensure the model and manifest describe the same inference schema."""

    artifact_features = _validated_name_sequence(
        artifact["feature_columns"],
        "artifact.feature_columns",
        allow_empty=False,
    )
    manifest_features = _validated_name_sequence(
        manifest["expected_feature_columns"],
        "manifest.expected_feature_columns",
        allow_empty=False,
    )
    if manifest_features != artifact_features:
        raise ValueError(
            "Training manifest and model artifact expect different feature "
            "schemas. Use artifacts from the same training run."
        )
    if not isinstance(artifact["target_column"], str) or not artifact[
        "target_column"
    ].strip():
        raise ValueError("Model target_column must be a non-empty string.")
    if not isinstance(manifest["target_column"], str) or not manifest[
        "target_column"
    ].strip():
        raise ValueError("Manifest target_column must be a non-empty string.")
    if manifest["target_column"] != artifact["target_column"]:
        raise ValueError(
            "Training manifest and model artifact use different targets."
        )

    numeric_columns = _validated_name_sequence(
        artifact["numeric_columns"],
        "artifact.numeric_columns",
    )
    categorical_columns = _validated_name_sequence(
        artifact["categorical_columns"],
        "artifact.categorical_columns",
    )
    numeric_set = set(numeric_columns)
    categorical_set = set(categorical_columns)
    feature_set = set(artifact_features)
    if numeric_set & categorical_set:
        raise ValueError(
            "Model columns cannot be both numeric and categorical."
        )
    typed_columns = numeric_set | categorical_set
    if typed_columns != feature_set:
        missing_typed_columns = sorted(feature_set - typed_columns)
        unknown_typed_columns = sorted(typed_columns - feature_set)
        raise ValueError(
            "Model numeric_columns and categorical_columns must form an "
            "exact partition of feature_columns; "
            f"untyped={missing_typed_columns}, "
            f"unknown={unknown_typed_columns}."
        )

    for field_name, artifact_values in (
        ("numeric_columns", numeric_columns),
        ("categorical_columns", categorical_columns),
    ):
        if field_name in manifest:
            manifest_values = _validated_name_sequence(
                manifest[field_name],
                f"manifest.{field_name}",
            )
            if manifest_values != artifact_values:
                raise ValueError(
                    f"Training manifest and model artifact have different "
                    f"{field_name}."
                )

    artifact_classes = _validated_classes(
        artifact["classes"],
        "artifact.classes",
    )
    if "classes" in manifest:
        manifest_classes = _validated_classes(
            manifest["classes"],
            "manifest.classes",
        )
        if list(manifest_classes) != list(artifact_classes):
            raise ValueError(
                "Training manifest and model artifact have different classes."
            )

    if "model_type" in manifest:
        manifest_model_type = manifest["model_type"]
        artifact_model_type = artifact.get("model_type")
        if not isinstance(manifest_model_type, str) or not (
            manifest_model_type.strip()
        ):
            raise ValueError("Manifest model_type must be a non-empty string.")
        if not isinstance(artifact_model_type, str) or not (
            artifact_model_type.strip()
        ):
            raise ValueError(
                "Model artifact is missing a valid model_type required by "
                "the manifest."
            )
        if manifest_model_type != artifact_model_type:
            raise ValueError(
                "Training manifest and model artifact have different "
                "model_type values."
            )

    if "random_state" in manifest:
        manifest_random_state = manifest["random_state"]
        artifact_random_state = artifact.get("random_state")
        if isinstance(manifest_random_state, bool) or not isinstance(
            manifest_random_state,
            Integral,
        ):
            raise ValueError("Manifest random_state must be an integer.")
        if isinstance(artifact_random_state, bool) or not isinstance(
            artifact_random_state,
            Integral,
        ):
            raise ValueError(
                "Model artifact is missing a valid random_state required by "
                "the manifest."
            )
        if int(manifest_random_state) != int(artifact_random_state):
            raise ValueError(
                "Training manifest and model artifact have different "
                "random_state values."
            )

    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "predict"):
        raise ValueError("Model artifact pipeline does not support prediction.")
    model_classes = getattr(pipeline, "classes_", None)
    if model_classes is None:
        raise ValueError("Model artifact pipeline does not expose classes_.")
    if list(model_classes) != list(artifact_classes):
        raise ValueError(
            "Model class order does not match the saved artifact classes."
        )

    # Validate deterministic feature rules when resources are loaded so an
    # application fails before accepting a request.
    _feature_config_from_manifest(manifest)

    recorded_model = Path(str(manifest["model_path"])).expanduser().resolve()
    if recorded_model != model_path:
        LOGGER.warning(
            "Selected model path differs from the manifest's recorded path; "
            "schema compatibility was verified: %s",
            model_path,
        )


def _feature_config_from_manifest(
    manifest: dict[str, Any],
) -> FeatureBuildConfig:
    """Reconstruct deterministic feature rules for an in-memory request."""

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

    drop_columns = _validated_name_sequence(
        values["drop_columns"],
        "feature_build_config.drop_columns",
    )
    date_columns = _validated_name_sequence(
        values["date_columns"],
        "feature_build_config.date_columns",
    )
    age_from_year_columns = _validated_name_sequence(
        values["age_from_year_columns"],
        "feature_build_config.age_from_year_columns",
    )
    log_columns = _validated_name_sequence(
        values["log_columns"],
        "feature_build_config.log_columns",
    )
    ratio_features = _validated_ratio_features(values["ratio_features"])

    keep_date_columns = values["keep_date_columns"]
    if not isinstance(keep_date_columns, bool):
        raise ValueError(
            "feature_build_config.keep_date_columns must be a boolean."
        )
    if keep_date_columns:
        raise ValueError(
            "In-memory inference does not support "
            "feature_build_config.keep_date_columns=True because retained "
            "date dtypes can differ from the file-based training pipeline. "
            "Retrain with keep_date_columns=False or use file-based inference."
        )

    reference_year = values["reference_year"]
    if reference_year is not None and (
        isinstance(reference_year, bool)
        or not isinstance(reference_year, Integral)
    ):
        raise ValueError(
            "feature_build_config.reference_year must be an integer or null."
        )

    return FeatureBuildConfig(
        source_path=Path("streamlit_input.csv"),
        target_column=None,
        drop_columns=drop_columns,
        date_columns=date_columns,
        keep_date_columns=keep_date_columns,
        age_from_year_columns=age_from_year_columns,
        reference_year=(
            int(reference_year) if reference_year is not None else None
        ),
        log_columns=log_columns,
        ratio_features=ratio_features,
    )


def load_inference_resources(
    model_path: str | Path,
    manifest_path: str | Path,
) -> InferenceResources:
    """Load and validate a trusted model/manifest pair."""

    resolved_model = Path(model_path).expanduser().resolve()
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    artifact = load_model_artifact(resolved_model)
    manifest = load_training_manifest(resolved_manifest)
    _validate_resource_pair(artifact, manifest, resolved_model)
    return InferenceResources(
        model_path=resolved_model,
        manifest_path=resolved_manifest,
        artifact=artifact,
        manifest=manifest,
    )


def run_dataframe_inference(
    data: pd.DataFrame,
    resources: InferenceResources,
    *,
    input_stage: str = "cleaned",
    prediction_column: str | None = None,
    include_input: bool = True,
    include_probabilities: bool = True,
) -> DataFrameInferenceResult:
    """Apply saved feature rules and predict entirely in memory."""

    if input_stage not in {"cleaned", "processed"}:
        raise ValueError("input_stage must be 'cleaned' or 'processed'.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Inference input must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Inference dataset contains no rows.")
    if len(data.columns) == 0:
        raise ValueError("Inference dataset contains no columns.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    original = data.reset_index(drop=True).copy()
    engineered_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    if input_stage == "cleaned":
        feature_config = _feature_config_from_manifest(resources.manifest)
        features, engineered_features, warnings, _ = build_features(
            original,
            feature_config,
        )
        if len(features) != len(original):
            raise RuntimeError(
                "Inference feature building changed the number of rows."
            )
    else:
        features = original.copy()

    output_column = prediction_column or (
        f"predicted_{resources.artifact['target_column']}"
    )
    if include_input and output_column in original.columns:
        raise ValueError(
            f"Prediction column '{output_column}' already exists in input."
        )

    model_result = predict_dataframe(
        features,
        resources.artifact,
        prediction_column=output_column,
        include_input=False,
        include_probabilities=include_probabilities,
    )
    if include_input:
        result = original.copy()
        generated_columns = (
            model_result.prediction_column,
            *model_result.probability_columns,
        )
        for column in generated_columns:
            if column in result.columns:
                raise ValueError(
                    f"Generated column '{column}' already exists in input."
                )
            result[column] = model_result.data[column].to_numpy()
    else:
        result = model_result.data

    for warning in warnings:
        LOGGER.warning("In-memory feature warning: %s", warning)
    LOGGER.info(
        "In-memory inference completed: rows=%s, stage=%s",
        len(result),
        input_stage,
    )
    return DataFrameInferenceResult(
        data=result,
        features=features,
        resolved_input_stage=input_stage,
        prediction_column=model_result.prediction_column,
        probability_columns=model_result.probability_columns,
        engineered_features=engineered_features,
        warnings=warnings,
    )
