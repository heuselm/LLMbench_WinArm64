# =============================================================================
# setup.ps1 -- Create venv and install benchmark dependencies
# Run once before run_benchmark.ps1
#
# Usage (from PowerShell):
#   .\setup.ps1
#
# If you get "execution of scripts is disabled", run this first:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# =============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

Set-Location $ScriptDir

Write-Host "============================================================"
Write-Host "  LM Benchmark Setup"
Write-Host "  Machine: MH-IPS5x | Windows 11 ARM64"
Write-Host "============================================================"
Write-Host ""

# Check Python
try {
    $pyver = python --version 2>&1
    Write-Host "  Found: $pyver"
} catch {
    Write-Error "ERROR: Python not found in PATH. Install Python 3.10+ and retry."
    exit 1
}

# [1/3] Create venv
Write-Host "[1/3] Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "  venv already exists, skipping."
} else {
    python -m venv venv
    Write-Host "  Created venv\"
}

$PY  = "$ScriptDir\venv\Scripts\python.exe"
$PIP = "$ScriptDir\venv\Scripts\pip.exe"

# [2/3] Install packages
Write-Host "[2/3] Installing packages (openai, requests)..."
Write-Host "  Both have pre-built ARM64 Windows wheels -- no compiler needed."
& $PIP install --upgrade pip --quiet
& $PIP install openai requests --quiet

# [3/3] Verify
Write-Host "[3/3] Verifying..."
& $PY -c "import openai, requests; print('  openai', openai.__version__, '| requests', requests.__version__)"

Write-Host ""
Write-Host "============================================================"
Write-Host "  Setup complete!"
Write-Host ""
Write-Host "  NEXT STEPS:"
Write-Host "  1. Open LM Studio -> Local Server -> Start (port 1234)"
Write-Host "  2. .\run_benchmark.ps1"
Write-Host "============================================================"
