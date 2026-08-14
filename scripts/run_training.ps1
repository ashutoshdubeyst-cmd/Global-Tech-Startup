[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [string]$PythonPath,
    [string]$TargetColumn = "acquisition_status",
    [string]$ProcessedDirectory = "data/processed",
    [string]$ModelOutput = "models/startup_classifier.pkl",
    [string]$TrainingMetricsOutput = "reports/metrics/training_metrics.json",
    [string]$EvaluationMetricsOutput = "reports/metrics/evaluation_metrics.json",
    [string]$ManifestOutput = "models/training_pipeline_manifest.json",
    [string[]]$DropColumns = @("company_id"),
    [string[]]$DateColumns = @(),
    [switch]$KeepDateColumns,
    [string[]]$AgeFromYearColumns = @("founding_year"),
    [int]$ReferenceYear = 2026,
    [string[]]$LogColumns = @(
        "total_funding_usd_millions",
        "valuation_usd_millions",
        "revenue_arr_millions"
    ),
    [string[]]$RatioFeatures = @(
        "funding_per_employee=total_funding_usd_millions/current_headcount_2026",
        "valuation_to_revenue=valuation_usd_millions/revenue_arr_millions"
    ),
    [ValidateRange(0.000001, 1.0)]
    [double]$TrainFraction = 0.70,
    [ValidateRange(0.000001, 1.0)]
    [double]$ValidationFraction = 0.15,
    [ValidateRange(0.000001, 1.0)]
    [double]$TestFraction = 0.15,
    [int]$RandomState = 42,
    [switch]$NoStratify,
    [ValidateSet("csv", "parquet")]
    [string]$OutputFormat = "csv",
    [ValidateSet("logistic_regression", "random_forest")]
    [string]$ModelType = "logistic_regression",
    [int]$MaxIterations = 2000,
    [int]$NumberOfEstimators = 300,
    [switch]$NoClassWeight,
    [switch]$Overwrite
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$SourcePath = ConvertTo-ProjectPath $Source
Assert-ProjectFile $SourcePath "Cleaned training dataset"

$TrainingArguments = @(
    "-B",
    "-m", "src.churn_model.pipelines.training_pipeline",
    $SourcePath,
    "--target-column", $TargetColumn,
    "--processed-dir", (ConvertTo-ProjectPath $ProcessedDirectory),
    "--model-output", (ConvertTo-ProjectPath $ModelOutput),
    "--training-metrics-output", (ConvertTo-ProjectPath $TrainingMetricsOutput),
    "--evaluation-metrics-output", (ConvertTo-ProjectPath $EvaluationMetricsOutput),
    "--manifest-output", (ConvertTo-ProjectPath $ManifestOutput),
    "--train-fraction", (ConvertTo-InvariantString $TrainFraction),
    "--validation-fraction", (ConvertTo-InvariantString $ValidationFraction),
    "--test-fraction", (ConvertTo-InvariantString $TestFraction),
    "--random-state", "$RandomState",
    "--output-format", $OutputFormat,
    "--model-type", $ModelType,
    "--max-iter", "$MaxIterations",
    "--n-estimators", "$NumberOfEstimators"
)

if ($DropColumns.Count -gt 0) {
    $TrainingArguments += "--drop-columns"
    $TrainingArguments += $DropColumns
}
if ($DateColumns.Count -gt 0) {
    $TrainingArguments += "--date-columns"
    $TrainingArguments += $DateColumns
}
if ($KeepDateColumns) {
    $TrainingArguments += "--keep-date-columns"
}
if ($AgeFromYearColumns.Count -gt 0) {
    $TrainingArguments += "--age-from-year-columns"
    $TrainingArguments += $AgeFromYearColumns
    $TrainingArguments += @("--reference-year", "$ReferenceYear")
}
if ($LogColumns.Count -gt 0) {
    $TrainingArguments += "--log-columns"
    $TrainingArguments += $LogColumns
}
foreach ($RatioFeature in $RatioFeatures) {
    $TrainingArguments += @("--ratio-feature", $RatioFeature)
}
if ($NoStratify) {
    $TrainingArguments += "--no-stratify"
}
if ($NoClassWeight) {
    $TrainingArguments += "--no-class-weight"
}
if ($Overwrite) {
    $TrainingArguments += "--overwrite"
}

Push-Location -LiteralPath $ProjectRoot
try {
    Invoke-ProjectPython $ProjectPython $TrainingArguments
}
finally {
    Pop-Location
}
