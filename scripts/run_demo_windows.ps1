$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
try {
    $demoStatus = Invoke-RestMethod 'http://127.0.0.1:8765/api/status' -TimeoutSec 2
    if ($demoStatus.model -eq 'Gemma 3 4B' -and $demoStatus.checkpoint -eq 3900) {
        Write-Host 'Simile is already running: http://127.0.0.1:8765'
        return
    }
} catch {
    # No existing demo; start it in this terminal.
}
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
& .venv-windows/Scripts/python.exe demo/server.py
