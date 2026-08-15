"""Streamlit interface for the startup acquisition-status model."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.churn_model.data.cleaning import normalize_column_name
from src.churn_model.pipelines.dataframe_inference import (
    DataFrameInferenceResult,
    InferenceResources,
    load_inference_resources,
    run_dataframe_inference,
)
from src.logger import setup_logging


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "models" / "training_pipeline_manifest.json"
MODEL_CANDIDATES = (
    PROJECT_ROOT / "models" / "startup_classifier.pkl",
    PROJECT_ROOT / "models" / "startup_classifier.joblib",
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 10_000
MAX_UPLOAD_COLUMNS = 200
MISSING_OPTION = "Unknown / missing"

_SINGLE_RESULT_KEYS = ("single_prediction", "single_resource_signature")
_BATCH_RESULT_KEYS = (
    "batch_prediction",
    "batch_fingerprint",
    "batch_resource_signature",
)

APP_INPUT_COLUMNS = (
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
)

NONNEGATIVE_INPUT_COLUMNS = (
    "founding_year",
    "total_funding_usd_millions",
    "valuation_usd_millions",
    "revenue_arr_millions",
    "monthly_burn_rate_millions",
    "runway_months_2024",
    "peak_headcount_2023",
    "layoffs_2024_2025",
    "current_headcount_2026",
)

FALLBACK_CATEGORIES = {
    "domain": (
        "AI",
        "Biotech",
        "Cloud",
        "Commerce",
        "Data",
        "Energy",
        "Fintech",
        "Health",
        "Robotics",
        "Security",
    ),
    "country": ("France", "Germany", "India", "UK", "USA"),
    "city": (
        "Austin",
        "Bengaluru",
        "Berlin",
        "Boston",
        "Chicago",
        "Delhi",
        "Denver",
        "Hyderabad",
        "London",
        "Mumbai",
        "Munich",
        "New York",
        "Paris",
        "Pune",
    ),
    "funding_stage": ("Seed", "Series A", "Series B", "Series C"),
    "investor_tier": ("Tier 1", "Tier 2", "Tier 3"),
    "ai_adoption_level": ("High", "Medium", "Low"),
}


def _find_model_path() -> Path:
    """Select the trusted model filename recorded by the manifest."""

    if not MANIFEST_PATH.is_file():
        return MODEL_CANDIDATES[0]
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Training manifest could not be read: {MANIFEST_PATH}"
        ) from error
    if not isinstance(manifest, dict) or not manifest.get("model_path"):
        raise ValueError("Training manifest does not contain a model_path.")

    recorded_name = Path(str(manifest["model_path"])).name.casefold()
    matching = [
        candidate
        for candidate in MODEL_CANDIDATES
        if candidate.name.casefold() == recorded_name
    ]
    if not matching:
        allowed = [candidate.name for candidate in MODEL_CANDIDATES]
        raise ValueError(
            "Training manifest names an unsupported model file. "
            f"Expected one of: {allowed}"
        )
    return matching[0]


@st.cache_resource(show_spinner=False)
def _configure_logging() -> logging.Logger:
    """Configure the rotating project log once per app process."""

    return setup_logging(__name__, console=False)


@st.cache_resource(show_spinner=False)
def _cached_resources(
    model_path: str,
    model_stamp: tuple[int, int],
    manifest_path: str,
    manifest_stamp: tuple[int, int],
) -> InferenceResources:
    """Cache a trusted artifact pair and refresh it when either file changes."""

    del model_stamp, manifest_stamp
    return load_inference_resources(model_path, manifest_path)


def _load_resources() -> InferenceResources:
    """Resolve and cache the project's fixed server-side artifacts."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Training manifest was not found: {MANIFEST_PATH}"
        )
    model_path = _find_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {model_path}")
    model_stat = model_path.stat()
    manifest_stat = MANIFEST_PATH.stat()
    return _cached_resources(
        str(model_path),
        (model_stat.st_mtime_ns, model_stat.st_size),
        str(MANIFEST_PATH),
        (manifest_stat.st_mtime_ns, manifest_stat.st_size),
    )


def _resource_signature(resources: InferenceResources) -> str:
    """Return a stable signature for the loaded artifact pair."""

    model_stat = resources.model_path.stat()
    manifest_stat = resources.manifest_path.stat()
    payload = "|".join(
        (
            str(resources.model_path),
            str(model_stat.st_mtime_ns),
            str(model_stat.st_size),
            str(resources.manifest_path),
            str(manifest_stat.st_mtime_ns),
            str(manifest_stat.st_size),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clear_session_keys(keys: tuple[str, ...]) -> None:
    """Remove stored results without disturbing widget state."""

    for key in keys:
        st.session_state.pop(key, None)


def _synchronize_resource_state(signature: str) -> None:
    """Discard predictions created by an older model or manifest."""

    if st.session_state.get("active_resource_signature") != signature:
        _clear_session_keys(_SINGLE_RESULT_KEYS)
        _clear_session_keys(_BATCH_RESULT_KEYS)
        st.session_state["active_resource_signature"] = signature


def _show_operation_error(error: Exception, action: str) -> None:
    """Show safe errors while retaining diagnostics in the project log."""

    logger = logging.getLogger(__name__)
    if isinstance(error, (ValueError, FileNotFoundError)):
        logger.warning("%s rejected: %s", action, error)
        st.error(str(error))
        return
    logger.exception("%s failed.", action)
    st.error(
        f"{action} failed because of an internal error. "
        "See logs/churn_model.log for details."
    )


def _trained_categories(
    resources: InferenceResources,
) -> dict[str, tuple[str, ...]]:
    """Read categorical choices from the fitted one-hot encoder."""

    options = dict(FALLBACK_CATEGORIES)
    try:
        preprocessor = resources.artifact["pipeline"].named_steps["preprocessor"]
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["encoder"]
        categorical_columns = tuple(resources.artifact["categorical_columns"])
        for column, values in zip(
            categorical_columns,
            encoder.categories_,
            strict=True,
        ):
            cleaned_values = tuple(
                sorted(
                    {
                        str(value)
                        for value in values
                        if pd.notna(value) and str(value).strip()
                    }
                )
            )
            if cleaned_values:
                options[column] = cleaned_values
    except (AttributeError, KeyError, TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Could not extract fitted categories; using app defaults.",
            exc_info=True,
        )
    return options


def _generated_feature_names(manifest: dict[str, Any]) -> set[str]:
    """Return feature names created by the saved feature rules."""

    config = manifest["feature_build_config"]
    generated: set[str] = set()
    for column in config["date_columns"]:
        generated.update(
            {
                f"{column}_year",
                f"{column}_month",
                f"{column}_quarter",
                f"{column}_day_of_week",
            }
        )
    generated.update(
        f"{column}_age" for column in config["age_from_year_columns"]
    )
    generated.update(f"{column}_log1p" for column in config["log_columns"])
    for specification in config["ratio_features"]:
        feature_name, _ = str(specification).split("=", maxsplit=1)
        generated.add(feature_name.strip())
    return generated


def _required_cleaned_columns(
    resources: InferenceResources,
) -> tuple[str, ...]:
    """Derive cleaned-input columns required by the saved feature rules."""

    manifest = resources.manifest
    config = manifest["feature_build_config"]
    generated = _generated_feature_names(manifest)
    required = set(config["drop_columns"])
    required.update(config["date_columns"])
    required.update(config["age_from_year_columns"])
    required.update(config["log_columns"])
    for specification in config["ratio_features"]:
        _, expression = str(specification).split("=", maxsplit=1)
        numerator, denominator = expression.split("/", maxsplit=1)
        required.update({numerator.strip(), denominator.strip()})
    required.update(
        set(manifest["expected_feature_columns"]) - generated
    )
    required.discard(str(manifest["target_column"]))

    ordered = [column for column in APP_INPUT_COLUMNS if column in required]
    ordered.extend(sorted(required - set(ordered)))
    return tuple(ordered)


def _required_numeric_cleaned_columns(
    resources: InferenceResources,
) -> set[str]:
    """Derive numeric source columns required before feature engineering."""

    config = resources.manifest["feature_build_config"]
    generated = _generated_feature_names(resources.manifest)
    numeric = set(config["age_from_year_columns"])
    numeric.update(config["log_columns"])
    for specification in config["ratio_features"]:
        _, expression = str(specification).split("=", maxsplit=1)
        numerator, denominator = expression.split("/", maxsplit=1)
        numeric.update({numerator.strip(), denominator.strip()})
    numeric.update(set(resources.artifact["numeric_columns"]) - generated)
    return numeric & set(_required_cleaned_columns(resources))


def _category_widget(
    label: str,
    column: str,
    categories: dict[str, tuple[str, ...]],
    preferred: str,
    *,
    allow_missing: bool = False,
) -> object:
    """Render a learned-category selector with a custom-value override."""

    values = list(categories.get(column, ()))
    if preferred not in values:
        values.insert(0, preferred)
    if allow_missing:
        values.append(MISSING_OPTION)
    selected = st.selectbox(
        label,
        values,
        index=values.index(preferred),
        key=f"{column}_choice",
    )
    custom = st.text_input(
        f"Optional custom {label.lower()}",
        key=f"{column}_custom",
        help="When provided, this value overrides the selection above.",
    ).strip()
    if custom:
        return custom
    if selected == MISSING_OPTION:
        return pd.NA
    return selected


def _predict(
    data: pd.DataFrame,
    resources: InferenceResources,
) -> DataFrameInferenceResult:
    """Run the shared cleaned-data inference core and log only metadata."""

    logger = logging.getLogger(__name__)
    logger.info("App prediction started: rows=%s", len(data))
    result = run_dataframe_inference(
        data,
        resources,
        input_stage="cleaned",
        include_input=True,
        include_probabilities=True,
    )
    logger.info("App prediction completed: rows=%s", len(result.data))
    return result


def _prediction_chart(
    result: DataFrameInferenceResult,
    resources: InferenceResources,
) -> pd.DataFrame:
    """Create a class-probability table for one prediction."""

    labels = [str(value) for value in resources.artifact["classes"]]
    probabilities = [
        float(result.data.loc[0, column])
        for column in result.probability_columns
    ]
    return pd.DataFrame(
        {"class": labels, "probability": probabilities}
    ).set_index("class")


def _result_csv(data: pd.DataFrame) -> bytes:
    """Serialize results while neutralizing spreadsheet formulas."""

    export = data.copy()

    def safe_cell(value: object) -> object:
        if not isinstance(value, str):
            return value
        candidate = value.lstrip(" \t\r\n")
        if candidate.startswith(("=", "+", "-", "@")) or value.startswith(
            ("\t", "\r", "\n")
        ):
            return f"'{value}"
        return value

    for column in export.columns:
        export[column] = export[column].map(safe_cell)
    return export.to_csv(index=False).encode("utf-8")


def _read_uploaded_csv(
    payload: bytes,
    resources: InferenceResources,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Read and validate a cleaned CSV upload."""

    if not payload:
        raise ValueError("The uploaded CSV is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded CSV exceeds the 10 MB app limit.")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The CSV must use UTF-8 encoding.") from error

    try:
        raw_header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as error:
        raise ValueError("The CSV header could not be read.") from error
    if len(raw_header) > MAX_UPLOAD_COLUMNS:
        raise ValueError(
            f"The uploaded CSV has {len(raw_header):,} columns; the app limit "
            f"is {MAX_UPLOAD_COLUMNS:,}."
        )
    normalized_header = [normalize_column_name(value) for value in raw_header]
    blank_header_positions = [
        index + 1
        for index, value in enumerate(normalized_header)
        if not value
    ]
    if blank_header_positions:
        raise ValueError(
            "CSV column names cannot be blank. Blank header positions: "
            f"{blank_header_positions}"
        )
    header_counts = Counter(normalized_header)
    duplicate_headers = sorted(
        value for value, count in header_counts.items() if count > 1
    )
    if duplicate_headers:
        raise ValueError(
            f"Duplicate columns after normalization: {duplicate_headers}"
        )

    data = pd.read_csv(io.BytesIO(payload))
    if len(data.columns) != len(normalized_header):
        raise ValueError("The CSV header is malformed or inconsistent.")
    data.columns = normalized_header
    if data.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if len(data) > MAX_UPLOAD_ROWS:
        raise ValueError(
            f"The uploaded CSV has {len(data):,} rows; the app limit is "
            f"{MAX_UPLOAD_ROWS:,}."
        )
    empty_rows = data.isna().all(axis=1)
    if empty_rows.any():
        row_numbers = (data.index[empty_rows] + 2).tolist()[:10]
        raise ValueError(
            "Completely empty CSV rows are not allowed. Example line numbers: "
            f"{row_numbers}"
        )

    warnings: list[str] = []
    target_column = str(resources.artifact["target_column"])
    if target_column in data.columns:
        data = data.drop(columns=[target_column])
        warnings.append(
            f"Removed target column '{target_column}' from the upload."
        )

    required = set(_required_cleaned_columns(resources))
    missing = sorted(required - set(data.columns))
    if missing == ["company_id"]:
        data["company_id"] = [
            f"UPLOAD-{index + 1:05d}" for index in range(len(data))
        ]
        warnings.append("Generated company_id values for the uploaded rows.")
        missing = []
    if missing:
        raise ValueError(f"CSV is missing required cleaned columns: {missing}")

    extra = sorted(set(data.columns) - required)
    if extra:
        warnings.append(f"Extra columns will be preserved but ignored: {extra}")

    numeric_columns = _required_numeric_cleaned_columns(resources)
    invalid_numeric: dict[str, int] = {}
    non_finite_numeric: dict[str, list[str]] = {}
    for column in sorted(numeric_columns):
        original = data[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        if invalid_count:
            invalid_numeric[column] = invalid_count
        non_finite_mask = converted.notna() & ~np.isfinite(converted)
        if non_finite_mask.any():
            non_finite_numeric[column] = [
                f"line {int(index) + 2}: {original.loc[index]!r}"
                for index in original.index[non_finite_mask][:5]
            ]
        data[column] = converted
    if invalid_numeric:
        raise ValueError(
            "Invalid nonnumeric values were found in numeric columns: "
            f"{invalid_numeric}"
        )
    if non_finite_numeric:
        raise ValueError(
            "Infinite numeric values are not allowed. Examples by column: "
            f"{non_finite_numeric}"
        )

    for column in set(resources.artifact["categorical_columns"]) & set(
        data.columns
    ):
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )
    if "company_id" in data.columns:
        data["company_id"] = (
            data["company_id"]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    log_columns = set(
        resources.manifest["feature_build_config"]["log_columns"]
    )
    nonnegative_columns = (
        set(NONNEGATIVE_INPUT_COLUMNS) | log_columns
    ) & numeric_columns
    negative_values: dict[str, list[str]] = {}
    for column in sorted(nonnegative_columns):
        negative_mask = data[column].notna() & (data[column] < 0)
        if negative_mask.any():
            negative_values[column] = [
                f"line {int(index) + 2}: {data.loc[index, column]!r}"
                for index in data.index[negative_mask][:5]
            ]
    if negative_values:
        raise ValueError(
            "Negative values are not allowed in nonnegative startup fields. "
            f"Examples by column: {negative_values}"
        )
    return data.reset_index(drop=True), tuple(warnings)


def _render_single_prediction(
    resources: InferenceResources,
    categories: dict[str, tuple[str, ...]],
    resource_signature: str,
) -> None:
    """Render the single-startup form and result."""

    st.subheader("Predict one startup")
    st.caption(
        "Enter cleaned business values. Acquisition status is the model output "
        "and is therefore not requested."
    )

    with st.form("single_startup_form"):
        identity, finances, workforce = st.columns(3)
        with identity:
            company_id = st.text_input("Company ID", value="APP-001").strip()
            domain = _category_widget(
                "Domain", "domain", categories, "AI"
            )
            founding_year = st.number_input(
                "Founding year",
                min_value=1900,
                max_value=2026,
                value=2019,
                step=1,
            )
            country = _category_widget(
                "Country", "country", categories, "India"
            )
            city = _category_widget(
                "City", "city", categories, "Bengaluru"
            )

        with finances:
            funding_stage = _category_widget(
                "Funding stage",
                "funding_stage",
                categories,
                "Series B",
            )
            total_funding = st.number_input(
                "Total funding (USD millions)",
                min_value=0.0,
                value=10.75,
                step=0.25,
            )
            valuation = st.number_input(
                "Valuation (USD millions)",
                min_value=0.0,
                value=44.0,
                step=1.0,
            )
            revenue = st.number_input(
                "Revenue ARR (USD millions)",
                min_value=0.0,
                value=4.7,
                step=0.1,
            )
            burn_rate = st.number_input(
                "Monthly burn rate (USD millions)",
                min_value=0.0,
                value=0.46,
                step=0.01,
            )

        with workforce:
            runway = st.number_input(
                "Runway months (2024)",
                min_value=0,
                value=15,
                step=1,
            )
            peak_headcount = st.number_input(
                "Peak headcount (2023)",
                min_value=0,
                value=74,
                step=1,
            )
            layoffs = st.number_input(
                "Layoffs (2024–2025)",
                min_value=0,
                value=11,
                step=1,
            )
            current_headcount = st.number_input(
                "Current headcount (2026)",
                min_value=0,
                value=63,
                step=1,
            )
            investor_tier = _category_widget(
                "Investor tier",
                "investor_tier",
                categories,
                "Tier 1",
            )
            ai_adoption = _category_widget(
                "AI adoption",
                "ai_adoption_level",
                categories,
                "High",
                allow_missing=True,
            )

        submitted = st.form_submit_button(
            "Predict acquisition status",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        _clear_session_keys(_SINGLE_RESULT_KEYS)
        try:
            if not company_id:
                raise ValueError("Company ID cannot be empty.")
            categorical_values = {
                "domain": domain,
                "country": country,
                "city": city,
                "funding_stage": funding_stage,
                "investor_tier": investor_tier,
                "ai_adoption_level": ai_adoption,
            }
            empty_custom = [
                column
                for column, value in categorical_values.items()
                if not str(value).strip()
            ]
            if empty_custom:
                raise ValueError(
                    f"Custom categorical values cannot be empty: {empty_custom}"
                )

            input_data = pd.DataFrame(
                [
                    {
                        "company_id": company_id,
                        "domain": domain,
                        "founding_year": int(founding_year),
                        "country": country,
                        "city": city,
                        "funding_stage": funding_stage,
                        "total_funding_usd_millions": float(total_funding),
                        "valuation_usd_millions": float(valuation),
                        "revenue_arr_millions": float(revenue),
                        "monthly_burn_rate_millions": float(burn_rate),
                        "runway_months_2024": int(runway),
                        "peak_headcount_2023": int(peak_headcount),
                        "layoffs_2024_2025": int(layoffs),
                        "current_headcount_2026": int(current_headcount),
                        "investor_tier": investor_tier,
                        "ai_adoption_level": ai_adoption,
                    }
                ]
            )
            prediction = _predict(input_data, resources)
            st.session_state["single_prediction"] = prediction
            st.session_state["single_resource_signature"] = resource_signature
            if current_headcount == 0:
                st.warning(
                    "Funding per employee is missing because current "
                    "headcount is zero."
                )
            if layoffs > peak_headcount:
                st.warning("Layoffs are greater than peak headcount.")
        except Exception as error:
            _clear_session_keys(_SINGLE_RESULT_KEYS)
            _show_operation_error(error, "Single-startup prediction")

    prediction = st.session_state.get("single_prediction")
    matching_resources = (
        st.session_state.get("single_resource_signature")
        == resource_signature
    )
    if isinstance(prediction, DataFrameInferenceResult) and matching_resources:
        predicted_value = prediction.data.loc[0, prediction.prediction_column]
        st.success(f"Predicted acquisition status: {predicted_value}")
        if prediction.probability_columns:
            chart = _prediction_chart(prediction, resources)
            confidence = float(chart["probability"].max())
            metric, note = st.columns([1, 3])
            metric.metric("Highest model probability", f"{confidence:.1%}")
            note.info(
                "Model probability is not certainty. Use this result as a "
                "decision-support signal, not as an automated business decision."
            )
            st.bar_chart(chart)
        for warning in prediction.warnings:
            st.warning(warning)
        with st.expander("Prediction details"):
            st.dataframe(
                prediction.data,
                use_container_width=True,
                hide_index=True,
            )
        st.download_button(
            "Download this prediction",
            data=_result_csv(prediction.data),
            file_name="startup_prediction.csv",
            mime="text/csv",
        )


def _render_batch_prediction(
    resources: InferenceResources,
    resource_signature: str,
) -> None:
    """Render cleaned CSV upload, validation, prediction, and download."""

    st.subheader("Predict a cleaned CSV")
    required_columns = _required_cleaned_columns(resources)
    template = pd.DataFrame(columns=required_columns)
    st.download_button(
        "Download empty CSV template",
        data=_result_csv(template),
        file_name="startup_inference_template.csv",
        mime="text/csv",
    )
    st.caption(
        "Limit: 10 MB, 10,000 rows, and 200 columns. Headers may use mixed "
        "case, but values must represent cleaned numeric and categorical fields."
    )

    uploaded = st.file_uploader(
        "Upload cleaned startup data",
        type=("csv",),
        accept_multiple_files=False,
    )
    if uploaded is None:
        _clear_session_keys(_BATCH_RESULT_KEYS)
        return

    payload = uploaded.getvalue()
    fingerprint = hashlib.sha256(
        resource_signature.encode("utf-8") + b"\0" + payload
    ).hexdigest()
    previous_fingerprint = st.session_state.get("batch_fingerprint")
    if previous_fingerprint not in (None, fingerprint):
        _clear_session_keys(_BATCH_RESULT_KEYS)
    try:
        preview, upload_warnings = _read_uploaded_csv(payload, resources)
        st.write(f"Validated rows: {len(preview):,}")
        for warning in upload_warnings:
            st.warning(warning)
        st.dataframe(preview.head(25), use_container_width=True, hide_index=True)
    except Exception as error:
        _clear_session_keys(_BATCH_RESULT_KEYS)
        _show_operation_error(error, "CSV validation")
        return

    if st.button(
        "Predict uploaded rows",
        type="primary",
        use_container_width=True,
    ):
        _clear_session_keys(_BATCH_RESULT_KEYS)
        try:
            prediction = _predict(preview, resources)
            st.session_state["batch_prediction"] = prediction
            st.session_state["batch_fingerprint"] = fingerprint
            st.session_state["batch_resource_signature"] = resource_signature
        except Exception as error:
            _clear_session_keys(_BATCH_RESULT_KEYS)
            _show_operation_error(error, "Batch prediction")

    prediction = st.session_state.get("batch_prediction")
    matching_upload = (
        st.session_state.get("batch_fingerprint") == fingerprint
    )
    matching_resources = (
        st.session_state.get("batch_resource_signature")
        == resource_signature
    )
    if (
        isinstance(prediction, DataFrameInferenceResult)
        and matching_upload
        and matching_resources
    ):
        st.success(f"Predicted {len(prediction.data):,} startup rows.")
        distribution = (
            prediction.data[prediction.prediction_column]
            .value_counts()
            .rename_axis("predicted_class")
            .to_frame("rows")
        )
        st.bar_chart(distribution)
        for warning in prediction.warnings:
            st.warning(warning)
        st.dataframe(
            prediction.data.head(200),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download all predictions",
            data=_result_csv(prediction.data),
            file_name="startup_batch_predictions.csv",
            mime="text/csv",
        )


def _render_model_information(resources: InferenceResources) -> None:
    """Display non-sensitive metadata about the active trusted model."""

    st.subheader("Model information")
    artifact = resources.artifact
    manifest = resources.manifest
    first, second, third = st.columns(3)
    first.metric("Model type", str(artifact.get("model_type", "unknown")))
    second.metric("Target", str(artifact["target_column"]))
    third.metric("Classes", len(artifact["classes"]))

    st.write("Class labels")
    st.write([str(value) for value in artifact["classes"]])
    st.write("Saved feature-building configuration")
    st.json(manifest["feature_build_config"])
    if "test_metrics" in manifest:
        st.write("Recorded test metrics")
        st.json(manifest["test_metrics"])
    with st.expander("Expected processed model features"):
        st.write(list(artifact["feature_columns"]))


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(
        page_title="Startup Acquisition Predictor",
        page_icon="🚀",
        layout="wide",
    )
    _configure_logging()

    st.title("Startup Acquisition Predictor")
    st.caption(
        "Interactive inference using the project's trusted model and saved "
        "feature rules."
    )
    try:
        resources = _load_resources()
        resource_signature = _resource_signature(resources)
        _synchronize_resource_state(resource_signature)
    except Exception as error:
        _show_operation_error(error, "Model initialization")
        st.code(
            ".\\scripts\\run_training.ps1 "
            "-Source \"tests\\test_data\\startups_cleaned.csv\""
        )
        st.info(
            "Train the model first, then refresh this page. Never upload or "
            "load an untrusted pickle/joblib model."
        )
        st.stop()

    st.sidebar.success("Model ready")
    st.sidebar.caption(f"Model: {resources.model_path.name}")
    st.sidebar.caption(f"Manifest: {resources.manifest_path.name}")
    st.sidebar.warning(
        "Demonstration model only. Predictions require human review."
    )

    required = set(_required_cleaned_columns(resources))
    categories = _trained_categories(resources)
    single_tab, batch_tab, model_tab = st.tabs(
        ("Single startup", "Batch CSV", "Model information")
    )
    with single_tab:
        unsupported_required = sorted(
            required - set(APP_INPUT_COLUMNS)
        )
        if not unsupported_required:
            _render_single_prediction(
                resources,
                categories,
                resource_signature,
            )
        else:
            st.warning(
                "The active model requires fields that are not available in "
                f"the fixed form: {unsupported_required}. Use the Batch CSV "
                "tab and its generated template."
            )
    with batch_tab:
        _render_batch_prediction(resources, resource_signature)
    with model_tab:
        _render_model_information(resources)


if __name__ == "__main__":
    main()
