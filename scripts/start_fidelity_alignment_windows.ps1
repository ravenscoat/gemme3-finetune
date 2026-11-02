param([ValidateSet('alignment','schema_pilot','prefix_pilot','prefix_alignment','consistency')][string]$Experiment = 'alignment')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. ./scripts/env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
if ((Test-Path reports/assessment_gpu.lock) -or (Test-Path reports/multi_dataset_gpu.lock)) { throw 'GPU is reserved; inspect owner.' }
$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
$demo = if ($listener) { Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 10 } else { $null }
if ($demo.engine.state -in @('loading','generating')) { throw 'The demo is actively using GPU.' }
if ($null -ne $demo.loaded_dataset) {
    $listener = Get-NetTCPConnection -LocalPort 8765 -State Listen
    $owned = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if (-not $owned.ExecutablePath.StartsWith('D:\gemma-finetune\') -or $owned.CommandLine -notmatch 'demo[/\\]server\.py') { throw 'Not our demo server.' }
    Stop-Process -Id $owned.ProcessId
    $replacement = Start-Process '.venv-windows/Scripts/python.exe' -ArgumentList '-u','demo/server.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'reports/demo_stdout.log' -RedirectStandardError 'reports/demo_stderr.log' -PassThru
    $replacement.Id | Set-Content reports/demo.pid
}
$taskName = if ($Experiment -eq 'consistency') { 'reliability_consistency' } elseif ($Experiment -eq 'prefix_alignment') { 'prefix_alignment' } elseif ($Experiment -eq 'prefix_pilot') { 'prefix_prompt_pilot' } elseif ($Experiment -eq 'schema_pilot') { 'schema_prompt_pilot' } else { 'fidelity_alignment' }
$scriptFile = if ($Experiment -eq 'consistency') { 'scripts/reliability_consistency_trial.py' } elseif ($Experiment -eq 'prefix_alignment') { 'scripts/run_prefix_alignment.py' } elseif ($Experiment -eq 'prefix_pilot') { 'scripts/prefix_prompt_pilot.py' } elseif ($Experiment -eq 'schema_pilot') { 'scripts/schema_prompt_pilot.py' } else { 'scripts/run_fidelity_alignment_queue.py' }
$worker = Start-Process '.venv-windows/Scripts/python.exe' -ArgumentList '-u',$scriptFile -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput "reports/${taskName}_stdout.log" -RedirectStandardError "reports/${taskName}_stderr.log" -PassThru
$worker.Id | Set-Content "reports/${taskName}_launcher.pid"
Write-Host "$taskName started. CMD monitor: scripts/watch_fidelity_cmd.cmd"
