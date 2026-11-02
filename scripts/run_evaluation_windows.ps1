$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
& .venv-windows/Scripts/python.exe scripts/generate_comparison.py @args
if ($LASTEXITCODE -ne 0) { throw 'Generation failed; completed outputs are retained for resume.' }
