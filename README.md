# Gemma 3 fine-tuning

Source code for training Gemma 3 4B QLoRA adapters and evaluating synthetic records across Bitext insurance support, InsuranceQA, reliability workflows, and MIMIC-IV demo data.

## Included

- `scripts/train_bitext.py`: Bitext QLoRA training.
- `scripts/train_dataset.py`: InsuranceQA, reliability, and MIMIC demo training.
- `scripts/train_mimic_conditioned.py`: conditioned MIMIC training.
- Data download/preparation, evaluation, analysis, and experiment scripts.
- Windows PowerShell runners and historical Linux/WSL setup scripts.
- Local demo in `demo/`, unit tests in `tests/`, and dependency lists.

## Windows setup and training

Use a CUDA-capable NVIDIA GPU. The Windows setup script expects a local uv binary at `.tools/uvwin/bin/uv.exe`; install uv there or adjust `$taskUv` before running it, and create the `reports/` directory for its environment snapshot. Local wheel caches are optional when packages are available online. Review `scripts/setup_windows.ps1` and `requirements-windows.txt` for the native Windows environment. The scripts store downloaded models, caches, and outputs within the project directory. Obtain Hugging Face access to the gated Gemma model and authenticate locally before downloading it; never commit access tokens.

```powershell
.\scripts\setup_windows.ps1
python scripts/download_bitext.py
python scripts/prepare_bitext.py
python scripts/download_training_model.py
.\scripts\run_training_windows.ps1 --run bitext-4b-pilot --steps 200 --samples 64
```

Use the configured environment's Python interpreter for preparation/download commands (or activate it first). Start with a small feasibility run before a full experiment. See the training scripts' `--help` for resume and dataset options.

For other datasets, inspect `scripts/download_remaining_sources.py` and `scripts/prepare_remaining_datasets.py`; source access and prerequisites vary. Dataset licenses and access requirements still apply.

## Evaluation and demo

Evaluation scripts consume locally prepared data and generated training artifacts. The demo additionally expects saved adapters, evaluation cohorts, and reports from completed runs:

```powershell
.\scripts\run_demo_windows.ps1
```

Open http://127.0.0.1:8765/ after the server starts. A fresh source checkout does not include the artifacts needed to populate the demo.

## Source-only repository

Datasets, downloaded base weights, trained adapters, checkpoints, generated reports, presentation assets, caches, virtual environments, and compiled Unsloth modules are excluded. Some analysis, reporting, and experiment scripts refer to those local artifacts or the original Windows workspace and require corresponding inputs or path configuration. Historical or experimental scripts are retained for reference; their presence does not imply they are the selected production policy.

Training learns structured synthetic records. Automated distribution and formatting metrics do not establish semantic, insurance, or clinical correctness.