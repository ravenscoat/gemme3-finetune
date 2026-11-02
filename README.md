# Gemma 3 · Synthetic Data Fine-Tuning

**Fine-tune Gemma 3 4B to generate structured synthetic records, then compare the results against the source data.**

This project brings together data preparation, QLoRA training, base-versus-adapter evaluation, and a local review dashboard. It explores four datasets spanning insurance support, question answering, reliability workflows, and clinical demo records.

The focus is on making the experiments inspectable: preserve split identities, record training protocols, compare matched generation settings, and examine formatting, distributions, repetition, and copying alongside qualitative review.

## What the project covers

- **Parameter-efficient training:** 4-bit Gemma 3 with QLoRA adapters using Unsloth, Transformers, and PyTorch.
- **Structured generation:** train on complete records and check generated JSON against dataset-specific schemas.
- **Data preparation:** download sources, construct training/validation/test splits, and save preparation audits.
- **Comparative evaluation:** inspect base and tuned generations with formatting, distribution, similarity, and review tools.
- **Local exploration:** browse source records, generate examples, and inspect saved results through the demo.
- **Experiment traceability:** retain seeds, input hashes, protocols, checkpoints, and training summaries in local run artifacts.

## Datasets

| Dataset | Task |
| --- | --- |
| Bitext insurance support | Generate support records with requests, replies, intents, and categories. |
| InsuranceQA | Generate structured insurance question-and-answer records. |
| Insurance AI reliability benchmark | Model structured reliability and workflow records. |
| MIMIC-IV demo | Explore structured clinical demo records, including conditioned generation experiments. |

Each dataset has its own preparation and evaluation path. Source licenses and access requirements apply independently.

## Training and evaluation workflow

```text
Download sources → Prepare and audit splits → Download base model
                                              ↓
                                    Train a QLoRA adapter
                                              ↓
                              Generate base/tuned comparisons
                                              ↓
                            Evaluate metrics and review examples
```

Training and evaluation outputs are saved locally. The repository contains the source code; saved weights and experiment results are separate artifacts.

## Getting started on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/ravenscoat/gemme3-finetune.git
cd gemme3-finetune
```

### 2. Prepare the environment

The native Windows scripts use Python 3.12 and a CUDA-capable NVIDIA GPU. Review [requirements-windows.txt](requirements-windows.txt) and [the setup script](scripts/setup_windows.ps1) before installation; the pinned GPU stack must suit your hardware and driver.

The setup script expects uv at `.tools/uvwin/bin/uv.exe`. Place your uv executable there or adjust `$taskUv` in the script to your installed uv path. It also writes an environment snapshot into `reports/`:

```powershell
New-Item -ItemType Directory -Force reports | Out-Null
.\scripts\setup_windows.ps1
```

Model downloads may require Hugging Face access and local authentication. Keep credentials out of the repository. The historical Linux/WSL scripts and [requirements-training.txt](requirements-training.txt) describe a separate environment.

### 3. Prepare Bitext and download the training model

Use the environment created by the setup script:

```powershell
.\.venv-windows\Scripts\python.exe scripts/download_bitext.py
.\.venv-windows\Scripts\python.exe scripts/prepare_bitext.py
.\.venv-windows\Scripts\python.exe scripts/download_training_model.py
```

The model downloader defaults to `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`, records its resolved revision, and stores files under `.cache/models/gemma3-4b-training/`.

### 4. Start with a feasibility run

```powershell
.\scripts\run_training_windows.ps1 --run bitext-smoke --steps 2 --samples 0 --train-limit 8 --eval-limit 8 --stress-test
```

Then run a bounded pilot using a separate run name:

```powershell
.\scripts\run_training_windows.ps1 --run bitext-pilot --steps 200 --samples 64
.\.venv-windows\Scripts\python.exe scripts/evaluate_bitext.py --run bitext-pilot
```

These commands are setup and pilot examples, not instructions to reproduce every historical experiment. Review script arguments and required inputs before launching longer jobs.

## Other datasets and experiments

| Entry point | Purpose |
| --- | --- |
| [download_remaining_sources.py](scripts/download_remaining_sources.py) | Download InsuranceQA and reliability sources. |
| [download_mimic_demo.py](download_mimic_demo.py) | Download MIMIC demo data. |
| [prepare_remaining_datasets.py](scripts/prepare_remaining_datasets.py) | Prepare the additional dataset tasks. |
| [train_bitext.py](scripts/train_bitext.py) | Train or resume the Bitext adapter. |
| [train_dataset.py](scripts/train_dataset.py) | Train an independent adapter for an additional dataset. |
| [train_mimic_conditioned.py](scripts/train_mimic_conditioned.py) | Run conditioned MIMIC training. |
| [generate_comparison.py](scripts/generate_comparison.py) | Generate comparison cohorts. |
| [evaluate_final_test.py](scripts/evaluate_final_test.py) | Evaluate a configured final-test run. |
| [distribution_benchmark.py](scripts/distribution_benchmark.py) | Run broader distribution analysis. |

Use training and evaluation scripts' `--help` where available. Additional evaluation libraries are listed in [requirements-evaluation.txt](requirements-evaluation.txt) and [requirements-analysis.txt](requirements-analysis.txt).

## Local demo

The demo provides a browser interface for generation, source-data exploration, and saved evaluation results.

```powershell
.\scripts\run_demo_windows.ps1
```

Open **http://127.0.0.1:8765/** after the server starts.

**The dashboard requires local data, adapters, and evaluation artifacts.** A fresh clone contains the application source but does not include the completed runs needed to populate every view. See [demo/README.md](demo/README.md) for interface details.

## Repository layout

```text
scripts/                      Preparation, training, evaluation, and experiment tools
demo/                         Local server and browser interface
tests/                        Unit tests for metrics and experiment helpers
requirements-windows.txt      Native Windows training dependencies
requirements-training.txt     Historical Linux/WSL training dependencies
requirements-evaluation.txt   Evaluation dependencies
requirements-analysis.txt     Additional analysis dependencies
```

Generated directories such as `data/`, `outputs/`, `reports/`, and `.cache/` are intentionally ignored, along with environments, model weights, and checkpoints.

## Scope and limitations

This is an experimental source release. Some reporting tools and historical experiments depend on local artifacts or paths from the original workspace and need configuration before use. No pretrained adapters, benchmark results, or complete reproducibility bundle are distributed here.

Formatting and distribution metrics describe specific aspects of generated data. They do not establish factual accuracy, insurance suitability, clinical validity, privacy guarantees, or equivalence to the source population. Qualitative and domain-expert review remain necessary for any downstream use.
