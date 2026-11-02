#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/gemma-finetune
runtime_root="$HOME/.local/share/gemma-finetune-20261030"
mkdir -p "$runtime_root"
if [ ! -x "$runtime_root/python/bin/python3.12" ]; then
  tar -xzf .tools/python-runtime.tar.gz -C "$runtime_root"
fi
if [ ! -x "$runtime_root/venv/bin/python" ]; then
  .tools/bootstrap/bin/uv venv --python "$runtime_root/python/bin/python3.12" "$runtime_root/venv"
fi
python3 scripts/prepare_native_runtime.py "$runtime_root"
"$runtime_root/venv/bin/python" -u -c 'import torch; print("Native runtime:",torch.__version__,torch.cuda.is_available()); x=torch.randn(32,32,device="cuda"); print((x@x).sum().item())'
