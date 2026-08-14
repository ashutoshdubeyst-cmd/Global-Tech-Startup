[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Data,

    [string]$PythonPath,
    [string]$Model = "models/startup_classifier.pkl",
    [string]$Manifest = "models/training_pipeline_manifest.json",
    [ValidateSet("auto", "cleaned", "processed")]
    [string]$InputStage = "cleaned",
    [string]$FeaturesOutput = "data/processed/inference_features.csv",
    [string]$Output = "reports/predictions/predictions.csv",
    [string]$PredictionColumn,
    [switch]$PredictionsOnly,
    [switch]$NoProbabilities,
    [switch]$DoNotPreserveOriginalColumns,
    [switch]$Overwrite
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$DataPath = ConvertTo-ProjectPath $Data
$ModelPath = ConvertTo-ProjectPath $Model
$ManifestPath = ConvertTo-ProjectPath $Manifest

Assert-ProjectFile $DataPath "Inference dataset"
Assert-ProjectFile $ModelPath "Trained model"
if ($InputStage -eq "cleaned") {
    Assert-ProjectFile $ManifestPath "Training manifest"
}

$InferenceArguments = @(
    "-B",
    "-m", "src.churn_model.pipelines.inference_pipeline",
    $DataPath,
    "--model", $ModelPath,
    "--manifest", $ManifestPath,
    "--input-stage", $InputStage,
    "--features-output", (ConvertTo-ProjectPath $FeaturesOutput),
    "--output", (ConvertTo-ProjectPath $Output)
)

if (-not [string]::IsNullOrWhiteSpace($PredictionColumn)) {
    $InferenceArguments += @("--prediction-column", $PredictionColumn)
}
if ($PredictionsOnly) {
    $InferenceArguments += "--predictions-only"
}
if ($NoProbabilities) {
    $InferenceArguments += "--no-probabilities"
}
if ($DoNotPreserveOriginalColumns) {
    $InferenceArguments += "--do-not-preserve-original-columns"
}
if ($Overwrite) {
    $InferenceArguments += "--overwrite"
}

Push-Location -LiteralPath $ProjectRoot
try {
    Invoke-ProjectPython $ProjectPython $InferenceArguments
}
finally {
    Pop-Location
}
