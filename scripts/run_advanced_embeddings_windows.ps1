$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. ./scripts/env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
if (Test-Path reports/assessment_gpu.lock) { throw 'GPU is already reserved; inspect its owner first.' }
$connections = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)
foreach ($connection in $connections) {
    $serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)"
    if (-not $serverProcess.ExecutablePath.StartsWith('D:\gemma-finetune\') -or $serverProcess.CommandLine -notmatch 'demo[/\\]server\.py') {
        throw 'Port8765 is owned by another process; refusing to stop it.'
    }
    Stop-Process -Id $serverProcess.ProcessId
}
$worker = Start-Process -FilePath '.venv-windows/Scripts/python.exe' -ArgumentList '-u','scripts/advanced_embeddings.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'reports/advanced_analysis/v1/embedding_stdout.log' -RedirectStandardError 'reports/advanced_analysis/v1/embedding_stderr.log' -PassThru
$worker.Id | Set-Content reports/advanced_analysis/v1/embedding_launcher.pid
$server = Start-Process -FilePath '.venv-windows/Scripts/python.exe' -ArgumentList '-u','demo/server.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'reports/demo_stdout.log' -RedirectStandardError 'reports/demo_stderr.log' -PassThru
$server.Id | Set-Content reports/demo.pid
Write-Host "Embedding worker started (launcher PID $($worker.Id)); demo restarted without a loaded model."
Write-Host 'Progress: reports/advanced_analysis/v1/embedding_progress.json'
