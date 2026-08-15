[CmdletBinding()]
param(
    [string]$PythonPath,
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [string]$Address = "localhost",
    [switch]$Headless
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$AppPath = ConvertTo-ProjectPath "app.py"
Assert-ProjectFile $AppPath "Streamlit application"

$AppArguments = @(
    "-B",
    "-m", "streamlit",
    "run", $AppPath,
    "--server.port", "$Port",
    "--server.address", $Address
)
if ($Headless) {
    $AppArguments += @("--server.headless", "true")
}

Push-Location -LiteralPath $ProjectRoot
try {
    Write-Host (
        "Starting the app at http://{0}:{1}" -f $Address, $Port
    )
    Write-Host "Press Ctrl+C to stop it."
    Invoke-ProjectPython $ProjectPython $AppArguments
}
finally {
    Pop-Location
}
