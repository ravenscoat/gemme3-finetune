$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
. .\scripts\env_windows.ps1
$taskUv = Join-Path $taskRoot '.tools/uvwin/bin/uv.exe'
& $taskUv python install 3.12 --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Windows Python setup failed.' }
if (-not (Test-Path -LiteralPath '.venv-windows/Scripts/python.exe')) {
    & $taskUv venv --python 3.12 .venv-windows
    if ($LASTEXITCODE -ne 0) { throw 'Windows virtual environment creation failed.' }
}
$taskPython = Join-Path $taskRoot '.venv-windows/Scripts/python.exe'
& $taskUv pip install --python $taskPython --find-links .tools/wheels --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 'torch==2.10.0+cu130' 'torchvision==0.25.0+cu130'
if ($LASTEXITCODE -ne 0) { throw 'Native CUDA PyTorch install failed.' }
& $taskPython -u -c "import torch; print('Windows torch:',torch.__version__,'CUDA:',torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(32,32,device='cuda'); print('GPU check:',(x@x).sum().item())"
if ($LASTEXITCODE -ne 0) { throw 'Windows GPU check failed.' }
# Install torchao from PyPI first, as recommended by the current Unsloth README.
& $taskUv pip install --python $taskPython torchao
if ($LASTEXITCODE -ne 0) { throw 'torchao installation failed.' }
& $taskUv pip install --python $taskPython --find-links .tools/wheels --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw 'Windows training libraries failed to install.' }
& $taskUv pip freeze --python $taskPython | Set-Content -LiteralPath 'reports/environment-windows-lock.txt' -Encoding utf8
& $taskPython -u -c "from unsloth import FastModel; print('Native Windows Unsloth import passed')"
if ($LASTEXITCODE -ne 0) { throw 'Windows Unsloth import failed.' }
