<#
.SYNOPSIS
    Start the FastAPI backend and the Vite frontend for development.

.DESCRIPTION
    Opens two background processes in the current console:
      * uvicorn on http://localhost:8000 (API + /docs)
      * Vite on http://localhost:5173 (proxies /api to the backend)
    Ctrl+C stops both.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv found. Run scripts\setup.ps1 first."
}
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies ..."
    Push-Location frontend
    npm install
    Pop-Location
}
if (-not (Test-Path "training\model\model-best")) {
    Write-Host "WARNING: training\model\model-best is missing; the API will use the"
    Write-Host "base en_core_web_sm NER, so the ablation is not meaningful."
    Write-Host "Train it with: $Python training\train_ner.py --epochs 30"
}

$Backend = Start-Process -FilePath $Python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" `
    -WorkingDirectory $Root -PassThru -NoNewWindow
$Frontend = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory (Join-Path $Root "frontend") -PassThru -NoNewWindow

Write-Host ""
Write-Host "Backend:  http://localhost:8000/docs"
Write-Host "Frontend: http://localhost:5173"
Write-Host "Press Ctrl+C to stop both."
Write-Host ""

try {
    while (-not $Backend.HasExited -and -not $Frontend.HasExited) {
        Start-Sleep -Seconds 2
    }
}
finally {
    foreach ($process in @($Backend, $Frontend)) {
        if ($process -and -not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
