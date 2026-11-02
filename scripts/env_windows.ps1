# Process-local paths: keep this task's environments, caches, and temp files on D:.
$taskRoot = Split-Path $PSScriptRoot -Parent
$env:UV_CACHE_DIR = Join-Path $taskRoot '.cache/uv-windows'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRoot '.tools/python-windows'
$env:UV_PYTHON_BIN_DIR = Join-Path $taskRoot '.tools/python-windows-bin'
$env:PIP_CACHE_DIR = Join-Path $taskRoot '.cache/pip-windows'
$env:HF_HOME = Join-Path $taskRoot '.cache/huggingface'
$env:TORCH_HOME = Join-Path $taskRoot '.cache/torch-windows'
$env:TORCH_EXTENSIONS_DIR = Join-Path $taskRoot '.cache/torch-extensions-windows'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $taskRoot '.cache/torchinductor-windows'
$env:TRITON_CACHE_DIR = Join-Path $taskRoot '.cache/triton-windows'
$env:TRITON_HOME = Join-Path $taskRoot '.cache/triton-home-windows'
$env:CUDA_CACHE_PATH = Join-Path $taskRoot '.cache/cuda-windows'
$env:UNSLOTH_DISABLE_STATISTICS = '1'
$env:XDG_CACHE_HOME = Join-Path $taskRoot '.cache'
$env:TEMP = Join-Path $taskRoot '.cache/tmp-windows'
$env:TMP = $env:TEMP
$env:UV_HTTP_TIMEOUT = '1800'
$env:UV_CONCURRENT_DOWNLOADS = '8'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:UV_CACHE_DIR,$env:UV_PYTHON_BIN_DIR | Out-Null
