#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/gemma-finetune
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export HF_HOME="$PWD/.cache/huggingface"
export TMPDIR="$PWD/.cache/tmp"
export UV_HTTP_TIMEOUT=1800
export UV_CONCURRENT_DOWNLOADS=8
mkdir -p "$TMPDIR" reports
if [ ! -x .tools/bootstrap/bin/uv ]; then
  python3 -m pip install --find-links .tools/wheels --target .tools/bootstrap uv
fi
UV="$PWD/.tools/bootstrap/bin/uv"
if [ ! -x .venv-wsl/bin/python ]; then
  if [ -x .tools/python-local/python/bin/python3.12 ]; then
    "$UV" venv --python .tools/python-local/python/bin/python3.12 .venv-wsl
  else
    "$UV" python install 3.12
    "$UV" venv --python 3.12 .venv-wsl
  fi
fi
"$UV" pip install --python .venv-wsl/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-training.txt
"$UV" pip freeze --python .venv-wsl/bin/python > reports/environment-lock.txt
.venv-wsl/bin/python -c 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(512,512,device="cuda"); print("GPU matmul:",(x@x).mean().item())'
