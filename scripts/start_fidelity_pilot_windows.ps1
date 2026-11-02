param([switch]$ConditionsOnly)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. ./scripts/env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
if ((Test-Path reports/assessment_gpu.lock) -or (Test-Path reports/multi_dataset_gpu.lock)) { throw 'GPU reservation exists; inspect its owner first.' }
$demo = Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 10
if ($demo.engine.state -in @('loading','generating')) { throw 'Demo is actively using GPU; wait for its request.' }
if ($null -ne $demo.loaded_dataset) {
    $listener = Get-NetTCPConnection -LocalPort 8765 -State Listen
    $ownedServer = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if (-not $ownedServer.ExecutablePath.StartsWith('D:\gemma-finetune\') -or $ownedServer.CommandLine -notmatch 'demo[/\\]server\.py') { throw 'Not our demo process.' }
    Stop-Process -Id $ownedServer.ProcessId
    $replacement = Start-Process '.venv-windows/Scripts/python.exe' -ArgumentList '-u','demo/server.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'reports/demo_stdout.log' -RedirectStandardError 'reports/demo_stderr.log' -PassThru
    $replacement.Id | Set-Content reports/demo.pid
}
$workerArguments = @('-u','scripts/mimic_fidelity_pilot.py')
$logStem = 'reports/mimic_fidelity_pilot'
if ($ConditionsOnly) { $workerArguments += '--conditions-only'; $logStem = 'reports/mimic_fidelity_conditions_only' }
$worker = Start-Process '.venv-windows/Scripts/python.exe' -ArgumentList $workerArguments -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput ($logStem + '_stdout.log') -RedirectStandardError ($logStem + '_stderr.log') -PassThru
$worker.Id | Set-Content -LiteralPath ($logStem + '_launcher.pid')
Write-Host "Fidelity pilot started (launcher PID $($worker.Id))."
Write-Host 'VS Code task: Watch fidelity improvement pilot'
