"""Unit tests for dataset validation rules."""

from pathlib import Path

import pandas as pd
import pytest

from src.churn_model.data.validation import (
    DataValidationError,
    ValidationConfig,
    validate_dataframe,
)


def test_validate_dataframe_returns_report_for_valid_data() -> None:
    data = pd.DataFrame(
        {
            "company_id": ["S001", "S002"],
            "country": ["India", "USA"],
            "acquisition_status": ["Acquired", "Independent"],
        }
    )
    config = ValidationConfig(
        data_path=Path("startups.csv"),
        required_columns=("company_id", "country", "acquisition_status"),
        target_column="acquisition_status",
        allowed_target_values=("Acquired", "Independent"),
        fail_on_duplicate_rows=True,
    )

    report = validate_dataframe(data, config)

    assert report.row_count == 2
    assert report.column_count == 3
    assert report.duplicate_row_count == 0
    assert report.warnings == ()


def test_validate_dataframe_reports_all_detected_errors() -> None:
    data = pd.DataFrame(
        {
            "company_id": ["S001", "S001"],
            "acquisition_status": ["Unknown", None],
        }
    )
    config = ValidationConfig(
        data_path=Path("startups.csv"),
        required_columns=("company_id", "country", "acquisition_status"),
        target_column="acquisition_status",
        allowed_target_values=("Acquired", "Independent"),
        fail_on_duplicate_rows=True,
    )

    with pytest.raises(DataValidationError) as error:
        validate_dataframe(data, config)

    message = str(error.value)
    assert "Missing required columns: ['country']" in message
    assert "contains 1 missing values" in message
    assert "unexpected values: ['Unknown']" in message