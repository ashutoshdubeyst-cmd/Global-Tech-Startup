# Project scripts

These PowerShell scripts are thin command wrappers around the Python modules in
`src/`. Run them from any directory; each script changes to the project root
before invoking Python. They use `venv`, then `.venv`, and finally a system
Python unless `-PythonPath` is supplied. Both standard Windows virtual
environments (`Scripts/python.exe`) and Conda prefix environments
(`python.exe` at the environment root) are supported.

If Windows blocks local scripts for the current terminal session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Recommended workflow

### 1. Create the environment

```powershell
.\scripts\setup_environment.ps1
```

To use a specific Anaconda interpreter:

```powershell
.\scripts\setup_environment.ps1 `
    -PythonExecutable "C:\Users\ashut\anaconda3\python.exe"
```

### 2. Ingest, validate, and clean labelled data

`-Source` must be a real CSV, JSON, Parquet, or Excel file. The script copies it
to `data/raw`, validates it, cleans it into `data/interim`, and validates the
cleaned result.

The defaults are a startup-data profile. They expect the columns used by this
project, including funding, valuation, revenue, headcount, investor, AI
adoption, and acquisition-status fields. Override the column-array parameters
when using a different schema.

```powershell
.\scripts\run_data_pipeline.ps1 `
    -Source "data\external\global_tech_startups_2026.csv"
```

If the file is already in `data/raw`, pass that path. Ingestion is then skipped:

```powershell
.\scripts\run_data_pipeline.ps1 `
    -Source "data\raw\global_tech_startups_2026.csv"
```

For unlabelled scoring data, use `-Unlabelled` so target validation is omitted:

```powershell
.\scripts\run_data_pipeline.ps1 `
    -Source "data\external\new_startups.csv" `
    -Unlabelled
```

This produces `data/interim/new_startups_cleaned.csv`, which can be supplied to
the inference script. Raw ingestion is intentionally retained if a later
validation or cleaning stage fails, so the source snapshot remains available
for investigation.

### 3. Train and evaluate

The training pipeline builds processed train, validation, and test datasets,
trains the model, evaluates the untouched test split, and writes the model,
metrics, and training manifest.

```powershell
.\scripts\run_training.ps1 `
    -Source "data\interim\global_tech_startups_2026_cleaned.csv"
```

Use `-ModelType random_forest` to select the other supported classifier.

When custom output paths are used, keep the model and manifest together:

```powershell
.\scripts\run_training.ps1 `
    -Source "data\interim\global_tech_startups_2026_cleaned.csv" `
    -ModelOutput "models\experiment_01.pkl" `
    -ManifestOutput "models\experiment_01_manifest.json"

.\scripts\run_inference.ps1 `
    -Data "data\interim\new_startups_cleaned.csv" `
    -Model "models\experiment_01.pkl" `
    -Manifest "models\experiment_01_manifest.json"
```

### 4. Generate predictions

Inference data must be cleaned with the same rules used for training. The
default `-InputStage cleaned` applies the saved feature-building rules without
refitting the model preprocessor.

```powershell
.\scripts\run_inference.ps1 `
    -Data "data\interim\new_startups_cleaned.csv"
```

For an already engineered table:

```powershell
.\scripts\run_inference.ps1 `
    -Data "data\processed\test.csv" `
    -InputStage processed
```

Only load model files that you created or otherwise trust.

### 5. Generate visualisations

The Python package uses the American module spelling `visualization`; the
wrapper filename uses `visualisation`.

```powershell
.\scripts\run_visualisation.ps1 `
    -Data "data\processed\train.csv" `
    -TargetColumn "acquisition_status"
```

### 6. Run tests

```powershell
.\scripts\run_tests.ps1
.\scripts\run_tests.ps1 -Suite unit
.\scripts\run_tests.ps1 -Suite integration
.\scripts\run_tests.ps1 -Keyword "validation" -VerboseOutput
```

Direct pytest runs use `work/pytest_tmp` as disposable storage. The test wrapper
uses a unique `tests/.pytest-run-*` directory and removes it after the run.
Never save permanent files in either location.

## Standalone feature building

Normally, call `run_training.ps1` because the training pipeline already builds
features. Use this script only when you need processed features without model
training:

```powershell
.\scripts\run_build_features.ps1 `
    -Source "data\interim\global_tech_startups_2026_cleaned.csv" `
    -TargetColumn "acquisition_status"
```

PowerShell array parameters use `@(...)`. For example:

```powershell
.\scripts\run_training.ps1 `
    -Source "data\interim\global_tech_startups_2026_cleaned.csv" `
    -DropColumns @("company_id", "company_name") `
    -RatioFeatures @(
        "funding_per_employee=total_funding_usd_millions/current_headcount_2026",
        "valuation_to_revenue=valuation_usd_millions/revenue_arr_millions"
    )
```

## Overwriting outputs

Outputs are protected by default. If a rerun should intentionally replace
existing files, add `-Overwrite` to that command. The scripts never delete
outputs themselves; overwrite checks remain inside the Python modules.

All Python stages write operational messages to `logs/churn_model.log`.
