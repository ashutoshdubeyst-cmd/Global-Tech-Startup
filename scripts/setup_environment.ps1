[CmdletBinding()]
param(
    [string]$PythonExecutable = "python",
    [string]$EnvironmentDirectory = "venv",
    [switch]$SkipPipUpgrade,
    [switch]$SkipDevelopmentDependencies
)

. (Join-Path $PSScriptRoot "_common.ps1")

Push-Location -LiteralPath $ProjectRoot
try {
    $EnvironmentPath = ConvertTo-ProjectPath $EnvironmentDirectory
    $EnvironmentPython = $null
    $EnvironmentPythonCandidates = @(
        (Join-Path $EnvironmentPath "Scripts\python.exe"),
        (Join-Path $EnvironmentPath "python.exe")
    )

    foreach ($Candidate in $EnvironmentPythonCandidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $EnvironmentPython = $Candidate
            break
        }
    }

    if (Test-Path -LiteralPath $EnvironmentPath) {
        if ($null -eq $EnvironmentPython) {
            throw (
                "The environment directory exists but contains neither " +
                "Scripts\python.exe (standard venv) nor python.exe " +
                "(Conda prefix): $EnvironmentPath"
            )
        }

        Write-Host "Using existing virtual environment: $EnvironmentPath"
    }
    else {
        $BootstrapCommand = Get-Command $PythonExecutable -ErrorAction SilentlyContinue
        if ($null -eq $BootstrapCommand) {
            $BootstrapCandidate = ConvertTo-ProjectPath $PythonExecutable
            if (-not (Test-Path -LiteralPath $BootstrapCandidate -PathType Leaf)) {
                throw "Python executable was not found: $PythonExecutable"
            }
            $BootstrapPython = $BootstrapCandidate
        }
        else {
            $BootstrapPython = $BootstrapCommand.Source
        }

        Write-Host "Creating virtual environment: $EnvironmentPath"
        & $BootstrapPython -m venv $EnvironmentPath
        if ($LASTEXITCODE -ne 0) {
            throw "Virtual-environment creation failed with exit code $LASTEXITCODE."
        }

        $EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $EnvironmentPython -PathType Leaf)) {
            throw (
                "The environment was created, but Python was not found at " +
                "$EnvironmentPython"
            )
        }
    }

    if (-not $SkipPipUpgrade) {
        Invoke-ProjectPython $EnvironmentPython @(
            "-m", "pip", "install", "--upgrade", "pip"
        )
    }

    $InstalledRequirements = $false
    $RuntimeRequirements = Join-Path $ProjectRoot "requirements.txt"
    $DevelopmentRequirements = Join-Path $ProjectRoot "requirements-dev.txt"

    if (Test-Path -LiteralPath $RuntimeRequirements -PathType Leaf) {
        Invoke-ProjectPython $EnvironmentPython @(
            "-m", "pip", "install", "-r", $RuntimeRequirements
        )
        $InstalledRequirements = $true
    }

    if (
        -not $SkipDevelopmentDependencies -and
        (Test-Path -LiteralPath $DevelopmentRequirements -PathType Leaf)
    ) {
        Invoke-ProjectPython $EnvironmentPython @(
            "-m", "pip", "install", "-r", $DevelopmentRequirements
        )
        $InstalledRequirements = $true
    }

    if (-not $InstalledRequirements) {
        throw (
            "No applicable dependency file was found. Expected " +
            "requirements.txt or requirements-dev.txt."
        )
    }

    Write-Host ""
    Write-Host "Environment setup completed."
    Write-Host "Python: $EnvironmentPython"
    Write-Host "Activation is optional; the run scripts use this Python directly."
}
finally {
    Pop-Location
}
