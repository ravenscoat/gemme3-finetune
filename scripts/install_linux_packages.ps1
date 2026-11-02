$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
# Linux packages are imported by WSL Python, not by Windows Python.
$env:UV_HTTP_TIMEOUT = '300'
$env:UV_CONCURRENT_DOWNLOADS = '4'
& .tools/uvwin/bin/uv.exe pip install --target .venv-wsl/lib/python3.12/site-packages --python-platform x86_64-manylinux_2_28 --python-version 3.12 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw 'Linux training dependency installation failed.' }
