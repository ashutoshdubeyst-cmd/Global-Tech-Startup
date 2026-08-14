"""Unit tests for deterministic data-cleaning rules."""

from pathlib import Path

import pandas as pd

from src.churn_model.data.cleaning import CleaningConfig, clean_dataframe


def test_clean_dataframe_standardizes_and_removes_bad_rows() -> None:
    raw_data = pd.DataFrame(
        {
            "Company ID": [" S001 ", " S001 ", None],
            "AI Adoption Level": [" High ", "High", "null"],
            "Funding USD": ["$1,000", "$1,000", ""],
        }
    )
    config = CleaningConfig(
        source_path=Path("unused.csv"),
        numeric_columns=("funding_usd",),
    )

    cleaned, report = clean_dataframe(raw_data, config)

    assert list(cleaned.columns) == [
        "company_id",
        "ai_adoption_level",
        "funding_usd",
    ]
    assert len(cleaned) == 1
    assert cleaned.loc[0, "company_id"] == "S001"
    assert cleaned.loc[0, "ai_adoption_level"] == "High"
    assert cleaned.loc[0, "funding_usd"] == 1_000
    assert report.empty_rows_removed == 1
    assert report.duplicate_rows_removed == 1


def test_clean_dataframe_does_not_modify_input() -> None:
    raw_data = pd.DataFrame(
        {
            "Company ID": [" S001 "],
            "Country": [" India "],
        }
    )
    original = raw_data.copy(deep=True)

    clean_dataframe(
        raw_data,
        CleaningConfig(source_path=Path("unused.csv")),
    )

    pd.testing.assert_frame_equal(raw_data, original)
