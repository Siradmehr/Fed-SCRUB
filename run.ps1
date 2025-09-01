# simulation.ps1
param(
    [string]$ConfigFile = "config.yaml",
    [switch]$Debug,
    [string]$ExperimentName = ""
)

# Activate virtual environment
..\fedvenv\Scripts\activate

# Set experiment name if provided
if ($ExperimentName) {
    $env:EXPERIMENT_NAME = $ExperimentName
}

# Define environment variables
$env:EXP_ENV_DIR = ".\envs"

# Create log directory if it doesn't exist
if (-Not (Test-Path -Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs"
}

# Create log file names with experiment name if provided
$logTime = Get-Date -Format "yyyyMMdd-HHmmss"
$logPrefix = if ($ExperimentName) { "$ExperimentName-$logTime" } else { $logTime }
$outLog = ".\logs\$logPrefix.out"
$errLog = ".\logs\$logPrefix.err"

# Start logging
"Starting simulation at $(Get-Date)" | Tee-Object -FilePath $outLog
"Environment: $env:EXP_ENV_DIR" | Tee-Object -FilePath $outLog -Append
if ($ExperimentName) {
    "Experiment: $ExperimentName" | Tee-Object -FilePath $outLog -Append
}

# Check if config file exists
if ($ConfigFile -and -Not (Test-Path $ConfigFile)) {
    Write-Error "Config file not found: $ConfigFile"
    exit 1
}

# OPTIONAL: Git update (uncomment if needed)
<#
Write-Output "Updating git repository..." | Tee-Object -FilePath $outLog -Append
git fetch
$CURRENT_BRANCH = git rev-parse --abbrev-ref HEAD
Write-Output "Current branch: $CURRENT_BRANCH" | Tee-Object -FilePath $outLog -Append
git pull origin $CURRENT_BRANCH | Tee-Object -FilePath $outLog -Append
Write-Output "Git repository updated to latest commit" | Tee-Object -FilePath $outLog -Append
#>

# Prepare python command
$pythonCmd = "python -m simulate"
if ($ConfigFile) {
    $pythonCmd += " --config $ConfigFile"
}
if ($Debug) {
    $pythonCmd += " --debug"
    $env:PYTHONPATH = "."
}

Write-Output "Running: $pythonCmd" | Tee-Object -FilePath $outLog -Append

# Run the simulation with error handling
try {
    Invoke-Expression "$pythonCmd 2>> `"$errLog`"" | Tee-Object -FilePath $outLog -Append
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        "Simulation completed successfully at $(Get-Date)" | Tee-Object -FilePath $outLog -Append
    } else {
        "Simulation failed with exit code $exitCode at $(Get-Date)" | Tee-Object -FilePath $outLog -Append
        Write-Error "Simulation failed. Check $errLog for details."
    }
} catch {
    "Simulation error: $($_.Exception.Message)" | Tee-Object -FilePath $outLog -Append
    Write-Error "Simulation encountered an error: $($_.Exception.Message)"
    exit 1
}

# Optional: Display log file locations
Write-Output "Logs saved to:"
Write-Output "  Output: $outLog"
Write-Output "  Errors: $errLog"