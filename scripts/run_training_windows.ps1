$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:UNSLOTH_DISABLE_STATISTICS = '1'
$taskModel = Join-Path $taskRoot '.cache/models/gemma3-4b-training'
& .venv-windows/Scripts/python.exe scripts/train_bitext.py --model $taskModel @args
if ($LASTEXITCODE -ne 0) { throw 'Windows training run failed; see its log.' }
