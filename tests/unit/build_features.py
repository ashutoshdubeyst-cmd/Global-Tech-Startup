"""Unit tests for feature engineering and deterministic splitting."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.churn_model.features.build_features import (
    FeatureBuildConfig,
    build_features,
    split_dataset,
)


def _feature_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_id": [f"S{index:03d}" for index in range(1, 9)],
            "founding_year": [2020, 2019, 2018, 2017, 2020, 2019, 2018, 2017],
            "total_funding": [1.0, 2.0, 4.0, 8.0, 1.5, 3.0, 6.0, 12.0],
            "headcount": [5, 10, 20, 40, 5, 10, 20, 40],
            "status": [
                "Independent",
                "Independent",
                "Acquired",
                "Acquired",
                "Independent",
                "Independent",
                "Acquired",
                "Acquired",
            ],
        }
    )


def _feature_config() -> FeatureBuildConfig:
    return FeatureBuildConfig(
        source_path=Path("unused.csv"),
        target_column="status",
        drop_columns=("company_id",),
        age_from_year_columns=("founding_year",),
        reference_year=2026,
        log_columns=("total_funding",),
        ratio_features=("funding_per_employee=total_funding/headcount",),
        train_fraction=0.50,
        validation_fraction=0.25,
        test_fraction=0.25,
        random_state=7,
    )


def test_build_features_creates_requested_columns() -> None:
    features, engineered, warnings, dropped_targets = build_features(
        _feature_source(),
        _feature_config(),
    )

    assert "company_id" not in features.columns
    assert "founding_year_age" in features.columns
    assert "total_funding_log1p" in features.columns
    assert "funding_per_employee" in features.columns
    assert features.loc[0, "founding_year_age"] == 6
    assert features.loc[0, "total_funding_log1p"] == np.log1p(1.0)
    assert features.loc[0, "funding_per_employee"] == 0.2
    assert set(engineered) == {
        "founding_year_age",
        "total_funding_log1p",
        "funding_per_employee",
    }
    assert warnings == ()
    assert dropped_targets == 0


def test_split_dataset_is_complete_and_reproducible() -> None:
    features, _, _, _ = build_features(_feature_source(), _feature_config())

    first, first_warning = split_dataset(features, _feature_config())
    second, second_warning = split_dataset(features, _feature_config())

    assert {name: len(data) for name, data in first.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert sum(len(data) for data in first.values()) == len(features)
    assert first_warning is None
    assert second_warning is None
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
