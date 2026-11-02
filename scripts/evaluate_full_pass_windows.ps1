$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:PYTHONUNBUFFERED = '1'
$comparisonRun = 'bitext-4b-full-pass-evaluation-500-windows'
$pilotComparison = 'bitext-4b-evaluation-500-windows'
& .venv-windows/Scripts/python.exe scripts/generate_comparison.py --source-run bitext-4b-full-pass-windows --run $comparisonRun --samples 500 --reuse-base-from $pilotComparison
if ($LASTEXITCODE -ne 0) { throw 'Full-pass generation failed; retain outputs for resume.' }
& .venv-windows/Scripts/python.exe scripts/evaluate_bitext.py --run $comparisonRun
if ($LASTEXITCODE -ne 0) { throw 'Strict evaluation failed.' }
& .venv-windows/Scripts/python.exe scripts/evaluate_bitext.py --run $comparisonRun --strip-code-fence
if ($LASTEXITCODE -ne 0) { throw 'Formatting-controlled evaluation failed.' }
& .venv-windows/Scripts/python.exe scripts/analyze_comparison.py --run $comparisonRun
if ($LASTEXITCODE -ne 0) { throw 'Extended evaluation failed.' }
& .venv-windows/Scripts/python.exe scripts/evaluate_mauve.py --run $comparisonRun --reference-run $pilotComparison
if ($LASTEXITCODE -ne 0) { throw 'MAUVE evaluation failed.' }
& .venv-windows/Scripts/python.exe scripts/compare_training_stages.py
if ($LASTEXITCODE -ne 0) { throw 'Three-stage comparison failed.' }
'Full-pass development assessment complete. Human review and final-test assessment remain pending.'
