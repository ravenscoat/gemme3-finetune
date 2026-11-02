$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
& .venv-windows/Scripts/python.exe -u scripts/run_remaining_queue.py
if ($LASTEXITCODE -ne 0) { throw 'Training queue stopped; inspect reports/remaining_training_status.json and its log.' }
