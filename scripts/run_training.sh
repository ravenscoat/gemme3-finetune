#!/usr/bin/env bash
# Historical WSL entry point. Native Windows is the user's selected runtime.
echo 'Use scripts/run_training_windows.ps1 from PowerShell; WSL training is retired.' >&2
exit 2
set -euo pipefail
cd /mnt/d/gemma-finetune
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# DrvFS directories report mode 0777; import helpers require private temp storage.
export TMPDIR
TMPDIR=$(mktemp -d /tmp/gemma-finetune.XXXXXX)
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor"
export PYTHONUNBUFFERED=1
export UNSLOTH_DISABLE_STATISTICS=1
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" reports
runtime_python="$PWD/.venv-wsl/bin/python"
if [ -f .tools/native-python.txt ]; then
  native_python=$(cat .tools/native-python.txt)
  if [ -x "$native_python" ]; then runtime_python="$native_python"; fi
fi
"$runtime_python" scripts/train_bitext.py --model "$PWD/.cache/models/gemma3-4b-training" "$@"
