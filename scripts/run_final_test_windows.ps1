$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
& .venv-windows/Scripts/python.exe scripts/evaluate_final_test.py
if ($LASTEXITCODE -ne 0) { throw 'Frozen test assessment failed; inspect final_test_errors.log.' }
