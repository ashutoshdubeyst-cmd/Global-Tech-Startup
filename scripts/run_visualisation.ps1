[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Data,

    [string]$PythonPath,
    [string]$OutputDirectory = "reports/figures",
    [string]$TargetColumn,
    [string[]]$NumericColumns = @(),
    [string[]]$CategoricalColumns = @(),
    [ValidateRange(1, 1000)]
    [int]$MaxNumericColumns = 12,
    [ValidateRange(1, 1000)]
    [int]$MaxCategoricalColumns = 6,
    [ValidateRange(1, 100000)]
    [int]$MaxCategories = 12,
    [ValidateRange(1, 100000)]
    [int]$HistogramBins = 30,
    [ValidateRange(50, 1200)]
    [int]$Dpi = 150,
    [switch]$Overwrite
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$DataPath = ConvertTo-ProjectPath $Data
Assert-ProjectFile $DataPath "Visualization dataset"

$VisualizationArguments = @(
    "-B",
    "-m", "src.churn_model.visualization.visualization",
    $DataPath,
    "--output-dir", (ConvertTo-ProjectPath $OutputDirectory),
    "--max-numeric-columns", "$MaxNumericColumns",
    "--max-categorical-columns", "$MaxCategoricalColumns",
    "--max-categories", "$MaxCategories",
    "--histogram-bins", "$HistogramBins",
    "--dpi", "$Dpi"
)

if (-not [string]::IsNullOrWhiteSpace($TargetColumn)) {
    $VisualizationArguments += @("--target-column", $TargetColumn)
}
if ($NumericColumns.Count -gt 0) {
    $VisualizationArguments += "--numeric-columns"
    $VisualizationArguments += $NumericColumns
}
if ($CategoricalColumns.Count -gt 0) {
    $VisualizationArguments += "--categorical-columns"
    $VisualizationArguments += $CategoricalColumns
}
if ($Overwrite) {
    $VisualizationArguments += "--overwrite"
}

Push-Location -LiteralPath $ProjectRoot
try {
    Invoke-ProjectPython $ProjectPython $VisualizationArguments
}
finally {
    Pop-Location
}
