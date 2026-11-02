$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
$evaluationRun = 'bitext-4b-evaluation-500-windows'
& .venv-windows/Scripts/python.exe scripts/generate_comparison.py --run $evaluationRun --samples 500
if ($LASTEXITCODE -ne 0) { throw 'Generation failed; resume preserves completed examples.' }
& .venv-windows/Scripts/python.exe scripts/evaluate_bitext.py --run $evaluationRun
if ($LASTEXITCODE -ne 0) { throw 'Strict evaluation failed.' }
& .venv-windows/Scripts/python.exe scripts/evaluate_bitext.py --run $evaluationRun --strip-code-fence
if ($LASTEXITCODE -ne 0) { throw 'Formatting-controlled evaluation failed.' }
& .venv-windows/Scripts/python.exe scripts/analyze_comparison.py --run $evaluationRun
if ($LASTEXITCODE -ne 0) { throw 'Extended evaluation failed.' }
'Generation and automated assessment complete. Human semantic review is still pending.'
