Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)

function ConvertTo-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if ([IO.Path]::IsPathRooted($PathValue)) {
        return [IO.Path]::GetFullPath($PathValue)
    }

    return [IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $PathValue)
    )
}

function Assert-ProjectFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,

        [string]$Description = "Input file"
    )

    if (-not (Test-Path -LiteralPath $PathValue -PathType Leaf)) {
        throw "$Description was not found: $PathValue"
    }
}

function Get-ProjectPython {
    param(
        [string]$PythonPath
    )

    if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
        $ProjectCandidate = ConvertTo-ProjectPath $PythonPath
        if (Test-Path -LiteralPath $ProjectCandidate -PathType Leaf) {
            return $ProjectCandidate
        }

        $ExplicitCommand = Get-Command $PythonPath -ErrorAction SilentlyContinue
        if ($null -ne $ExplicitCommand) {
            return $ExplicitCommand.Source
        }

        throw "The requested Python executable was not found: $PythonPath"
    }

    $Candidates = @(
        (Join-Path $ProjectRoot "venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot "venv\python.exe"),
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv\python.exe")
    )

    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return $Candidate
        }
    }

    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $SystemPython) {
        Write-Warning ((
                "No project virtual environment was found; using {0}. " +
                "Run scripts\setup_environment.ps1 to create one."
            ) -f $SystemPython.Source)
        return $SystemPython.Source
    }

    throw (
        "Python was not found. Run scripts\setup_environment.ps1 " +
        "or supply -PythonPath."
    )
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExecutable,

        [Parameter(Mandatory = $true)]
        [string[]]$PythonArguments
    )

    Write-Host "Running: $PythonExecutable $($PythonArguments -join ' ')"
    & $PythonExecutable @PythonArguments
    $NativeExitCode = $LASTEXITCODE

    if ($NativeExitCode -ne 0) {
        throw "Python failed with exit code $NativeExitCode."
    }
}

function ConvertTo-InvariantString {
    param(
        [Parameter(Mandatory = $true)]
        [double]$Value
    )

    return $Value.ToString(
        [Globalization.CultureInfo]::InvariantCulture
    )
}
