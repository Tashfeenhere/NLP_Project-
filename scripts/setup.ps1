<#
.SYNOPSIS
    Create the development environment for Clinical Note Analyzer.

.DESCRIPTION
    Creates .venv, installs pinned dependencies, downloads the spaCy English
    model, and generates the training corpus.  Pass -Train to also fine-tune
    the NER head (a few minutes on CPU).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Train
#>
[CmdletBinding()]
param(
    [switch]$Train
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv ..."
    py -3 -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
Write-Host "Installing Python dependencies ..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

Write-Host "Installing spaCy model en_core_web_sm ..."
& $Python -m spacy download en_core_web_sm

Write-Host "Generating synthetic corpus and DocBin files ..."
& $Python data\generate_notes.py --seed 42 --n 260
& $Python data\build_docbin.py
& $Python data\build_handwritten.py

if ($Train) {
    Write-Host "Fine-tuning the clinical NER head (CPU) ..."
    & $Python training\train_ner.py --epochs 30
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the app:  powershell -ExecutionPolicy Bypass -File scripts\dev.ps1"
Write-Host "Run tests:    .venv\Scripts\python.exe -m pytest"
