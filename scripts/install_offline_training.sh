#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/gemma-finetune
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR" reports
.tools/bootstrap/bin/uv pip install --python .venv-wsl/bin/python \
  --no-index --find-links .tools/wheels -r requirements-training.txt
.tools/bootstrap/bin/uv pip freeze --python .venv-wsl/bin/python > reports/environment-lock.txt
.venv-wsl/bin/python -c 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(512,512,device="cuda"); print("GPU matmul:",(x@x).mean().item())'
