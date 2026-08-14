[CmdletBinding()]
param(
    [string]$PythonPath,
    [ValidateSet("all", "unit", "integration")]
    [string]$Suite = "all",
    [string]$Keyword,
    [switch]$VerboseOutput,
    [switch]$StopOnFirstFailure
)

. (Join-Path $PSScriptRoot "_common.ps1")

$ProjectPython = Get-ProjectPython $PythonPath
$TestPath = switch ($Suite) {
    "unit" { "tests/unit" }
    "integration" { "tests/integration" }
    default { "tests" }
}

$TestRunId = [Guid]::NewGuid().ToString("N")
$ExpectedTestDirectory = ConvertTo-ProjectPath "tests"
$TestBaseTemp = ConvertTo-ProjectPath "tests/.pytest-run-$TestRunId"
if (
    -not [string]::Equals(
        [IO.Path]::GetDirectoryName($TestBaseTemp),
        $ExpectedTestDirectory,
        [StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "Refusing to use a test temp directory outside tests/: $TestBaseTemp"
}
$TestArguments = @(
    "-B",
    "-m", "pytest",
    $TestPath,
    "--basetemp", $TestBaseTemp
)
if (-not [string]::IsNullOrWhiteSpace($Keyword)) {
    $TestArguments += @("-k", $Keyword)
}
if ($VerboseOutput) {
    $TestArguments += "-v"
}
if ($StopOnFirstFailure) {
    $TestArguments += "-x"
}

Push-Location -LiteralPath $ProjectRoot
try {
    Invoke-ProjectPython $ProjectPython $TestArguments
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $TestBaseTemp) {
        try {
            Remove-Item -LiteralPath $TestBaseTemp -Recurse -Force
        }
        catch {
            Write-Warning "Could not remove test temp directory: $TestBaseTemp"
        }
    }
}
