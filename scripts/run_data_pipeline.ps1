[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [string]$PythonPath,
    [string]$RawDataDirectory = "data/raw",
    [string]$InterimDataDirectory = "data/interim",
    [string]$CleanedOutput,
    [string]$TargetColumn = "acquisition_status",

    [string[]]$RequiredColumns = @(
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
        "ai_adoption_level"
    ),

    [string[]]$AllowedTargetValues = @(),
    [ValidateRange(0.0, 1.0)]
    [double]$MaxMissingFraction = 1.0,

    [string[]]$NumericColumns = @(
        "founding_year",
        "total_funding_usd_millions",
        "valuation_usd_millions",
        "revenue_arr_millions",
        "monthly_burn_rate_millions",
        "runway_months_2024",
        "peak_headcount_2023",
        "layoffs_2024_2025",
        "current_headcount_2026"
    ),

    [string[]]$DateColumns = @(),
    [string[]]$MissingMarkers = @(),
    [switch]$Unlabelled,
    [switch]$FailOnDuplicates,
    [switch]$KeepDuplicates,
    [switch]$KeepEmptyRows,
    [switch]$Overwrite
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$SourcePath = ConvertTo-ProjectPath $Source
Assert-ProjectFile $SourcePath "Source dataset"

if ($FailOnDuplicates -and $KeepDuplicates) {
    throw "-FailOnDuplicates and -KeepDuplicates cannot be used together."
}
if ($Unlabelled -and $AllowedTargetValues.Count -gt 0) {
    throw "-AllowedTargetValues cannot be used together with -Unlabelled."
}

$RawDirectoryPath = ConvertTo-ProjectPath $RawDataDirectory
$RawPath = [IO.Path]::GetFullPath(
    (Join-Path $RawDirectoryPath ([IO.Path]::GetFileName($SourcePath)))
)

if ([string]::IsNullOrWhiteSpace($CleanedOutput)) {
    $InterimDirectoryPath = ConvertTo-ProjectPath $InterimDataDirectory
    $CleanedPath = [IO.Path]::GetFullPath(
        (Join-Path $InterimDirectoryPath (
            "$([IO.Path]::GetFileNameWithoutExtension($SourcePath))_cleaned.csv"
        ))
    )
}
else {
    $CleanedPath = ConvertTo-ProjectPath $CleanedOutput
}

$SourceIsRawDestination = [string]::Equals(
    $SourcePath,
    $RawPath,
    [StringComparison]::OrdinalIgnoreCase
)

if (
    [string]::Equals(
        $CleanedPath,
        $RawPath,
        [StringComparison]::OrdinalIgnoreCase
    ) -or
    [string]::Equals(
        $CleanedPath,
        $SourcePath,
        [StringComparison]::OrdinalIgnoreCase
    )
) {
    throw (
        "Cleaned output must differ from both the source and raw dataset: " +
        $CleanedPath
    )
}

if (
    -not $Overwrite -and
    -not $SourceIsRawDestination -and
    (Test-Path -LiteralPath $RawPath)
) {
    throw (
        "Raw output already exists: $RawPath. " +
        "Use -Overwrite only when replacement is intentional."
    )
}
if (-not $Overwrite -and (Test-Path -LiteralPath $CleanedPath)) {
    throw (
        "Cleaned output already exists: $CleanedPath. " +
        "Use -Overwrite only when replacement is intentional."
    )
}

$EffectiveRequiredColumns = @($RequiredColumns)
if ($Unlabelled) {
    $EffectiveRequiredColumns = @(
        $EffectiveRequiredColumns | Where-Object {
            -not [string]::Equals(
                $_,
                $TargetColumn,
                [StringComparison]::OrdinalIgnoreCase
            )
        }
    )
}

Push-Location -LiteralPath $ProjectRoot
try {
    if (-not $SourceIsRawDestination) {
        $IngestionArguments = @(
            "-B",
            "-m", "src.churn_model.data.ingestion",
            $SourcePath,
            "--raw-data-dir", $RawDirectoryPath
        )
        if ($Overwrite) {
            $IngestionArguments += "--overwrite"
        }
        Invoke-ProjectPython $ProjectPython $IngestionArguments
    }
    else {
        Write-Host "Source is already in data/raw; ingestion was skipped."
    }

    $ValidationArguments = @(
        "-B",
        "-m", "src.churn_model.data.validation",
        $RawPath,
        "--max-missing-fraction",
        (ConvertTo-InvariantString $MaxMissingFraction)
    )
    if ($FailOnDuplicates) {
        $ValidationArguments += "--fail-on-duplicates"
    }
    Invoke-ProjectPython $ProjectPython $ValidationArguments

    $CleaningArguments = @(
        "-B",
        "-m", "src.churn_model.data.cleaning",
        $RawPath,
        "--output", $CleanedPath
    )
    if ($NumericColumns.Count -gt 0) {
        $CleaningArguments += "--numeric-columns"
        $CleaningArguments += $NumericColumns
    }
    if ($DateColumns.Count -gt 0) {
        $CleaningArguments += "--date-columns"
        $CleaningArguments += $DateColumns
    }
    if ($MissingMarkers.Count -gt 0) {
        $CleaningArguments += "--missing-markers"
        $CleaningArguments += $MissingMarkers
    }
    if ($KeepDuplicates) {
        $CleaningArguments += "--keep-duplicates"
    }
    if ($KeepEmptyRows) {
        $CleaningArguments += "--keep-empty-rows"
    }
    if ($Overwrite) {
        $CleaningArguments += "--overwrite"
    }
    Invoke-ProjectPython $ProjectPython $CleaningArguments

    $CleanedValidationArguments = @(
        "-B",
        "-m", "src.churn_model.data.validation",
        $CleanedPath,
        "--max-missing-fraction",
        (ConvertTo-InvariantString $MaxMissingFraction)
    )
    if (-not $Unlabelled) {
        $CleanedValidationArguments += @("--target-column", $TargetColumn)
    }
    if ($EffectiveRequiredColumns.Count -gt 0) {
        $CleanedValidationArguments += "--required-columns"
        $CleanedValidationArguments += $EffectiveRequiredColumns
    }
    if (-not $Unlabelled -and $AllowedTargetValues.Count -gt 0) {
        $CleanedValidationArguments += "--allowed-target-values"
        $CleanedValidationArguments += $AllowedTargetValues
    }
    if ($FailOnDuplicates) {
        $CleanedValidationArguments += "--fail-on-duplicates"
    }
    Invoke-ProjectPython $ProjectPython $CleanedValidationArguments

    Write-Host ""
    Write-Host "Data pipeline completed."
    Write-Host "Raw dataset:     $RawPath"
    Write-Host "Cleaned dataset: $CleanedPath"
}
finally {
    Pop-Location
}
