$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\scripts\run_training_windows.ps1 `
    --run bitext-4b-full-pass-windows `
    --resume D:\gemma-finetune\outputs\bitext-4b-pilot-windows\checkpoints\checkpoint-200 `
    --steps 3900 --samples 0 --eval-limit 0 --eval-per-intent 10 `
    --checkpoint-every 500 --logging-every 25
if ($LASTEXITCODE -ne 0) { throw 'Full-pass training failed; check logs and saved checkpoints.' }
