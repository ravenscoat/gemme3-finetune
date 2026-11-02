$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:PYTHONUNBUFFERED = '1'
$generationPid = [int](Get-Content reports/evaluation500.pid)
while (Get-Process -Id $generationPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 5 }
if (-not (Test-Path reports/bitext-4b-evaluation-500-windows_assessment.md)) {
    throw 'Core assessment did not finish; MAUVE is not started.'
}
# Model downloads are permitted here, after the offline Gemma job has exited.
$env:HF_HUB_OFFLINE = '0'
& .venv-windows/Scripts/python.exe scripts/evaluate_mauve.py
if ($LASTEXITCODE -ne 0) { throw 'MAUVE evaluation failed; see its error log.' }
