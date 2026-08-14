[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [string]$PythonPath,
    [string]$OutputDirectory = "data/processed",
    [string]$TargetColumn,
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
    [ValidateRange(0.0, 1.0)]
    [double]$ValidationFraction = 0.15,
    [ValidateRange(0.0, 1.0)]
    [double]$TestFraction = 0.15,
    [int]$RandomState = 42,
    [switch]$NoStratify,
    [ValidateSet("csv", "parquet")]
    [string]$OutputFormat = "csv",
    [switch]$Overwrite
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$SourcePath = ConvertTo-ProjectPath $Source
$OutputDirectoryPath = ConvertTo-ProjectPath $OutputDirectory
Assert-ProjectFile $SourcePath "Cleaned feature source"

$FeatureArguments = @(
    "-B",
    "-m", "src.churn_model.features.build_features",
    $SourcePath,
    "--output-dir", $OutputDirectoryPath,
    "--train-fraction", (ConvertTo-InvariantString $TrainFraction),
    "--validation-fraction", (ConvertTo-InvariantString $ValidationFraction),
    "--test-fraction", (ConvertTo-InvariantString $TestFraction),
    "--random-state", "$RandomState",
    "--output-format", $OutputFormat
)

if (-not [string]::IsNullOrWhiteSpace($TargetColumn)) {
    $FeatureArguments += @("--target-column", $TargetColumn)
}
if ($DropColumns.Count -gt 0) {
    $FeatureArguments += "--drop-columns"
    $FeatureArguments += $DropColumns
}
if ($DateColumns.Count -gt 0) {
    $FeatureArguments += "--date-columns"
    $FeatureArguments += $DateColumns
}
if ($KeepDateColumns) {
    $FeatureArguments += "--keep-date-columns"
}
if ($AgeFromYearColumns.Count -gt 0) {
    $FeatureArguments += "--age-from-year-columns"
    $FeatureArguments += $AgeFromYearColumns
    $FeatureArguments += @("--reference-year", "$ReferenceYear")
}
if ($LogColumns.Count -gt 0) {
    $FeatureArguments += "--log-columns"
    $FeatureArguments += $LogColumns
}
foreach ($RatioFeature in $RatioFeatures) {
    $FeatureArguments += @("--ratio-feature", $RatioFeature)
}
if ($NoStratify) {
    $FeatureArguments += "--no-stratify"
}
if ($Overwrite) {
    $FeatureArguments += "--overwrite"
}

Push-Location -LiteralPath $ProjectRoot
try {
    Invoke-ProjectPython $ProjectPython $FeatureArguments
}
finally {
    Pop-Location
}
