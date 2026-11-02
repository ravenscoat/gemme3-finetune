$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. ./scripts/env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
if ((Test-Path reports/assessment_gpu.lock) -or (Test-Path reports/multi_dataset_gpu.lock)) { throw 'GPU is reserved; inspect owner.' }
$demo = Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 10
if ($demo.engine.state -in @('loading','generating')) { throw 'Demo currently uses GPU.' }
if ($null -ne $demo.loaded_dataset) { throw 'Unload the demo model before starting training.' }
$worker = Start-Process '.venv-windows/Scripts/python.exe' -ArgumentList '-u','scripts/run_mimic_conditioned_experiment.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'reports/mimic_conditioned_experiment_stdout.log' -RedirectStandardError 'reports/mimic_conditioned_experiment_stderr.log' -PassThru
$worker.Id | Set-Content reports/mimic_conditioned_experiment_launcher.pid
Write-Host "Candidate training and comparison queue started (launcher PID $($worker.Id))."
Write-Host 'CMD monitor: scripts/watch_fidelity_cmd.cmd'
