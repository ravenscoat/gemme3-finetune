# Simple app — current entry point

Open http://127.0.0.1:8765/ after running scripts/run_demo_windows.ps1.

- Generate: choose a dataset, edit the prompt and click Generate example. A collapsible training preview is beside the composer; saved examples are clearly labelled.
- Dataset: search, filter and page through source records.
- Results: evaluation counts and highlights first; typical-value comparisons and copying/review notes expand when needed. Distribution charts remain one click away.

Advanced settings, example review details and JSON/download controls are collapsed by default. The layout supports narrow screens and desktop columns. Screenshot: output/frontend/simplified-preview.png.

Reports are separate deliverables; the website contains no report hyperlinks.
Open /details#evaluation for the previous
detailed dashboard, or /evidence/criteria for the full distribution charts.
Saved examples come from the selected evaluation cohorts; live generation still
uses the original saved adapters/settings. MIMIC's selected candidate is not silently
deployed. This interface does not compute fresh evaluation scores for live outputs.

The sections below document the earlier detailed interface and completed jobs.

# Simile — Synthetic Data Studio

## Advanced evaluation added — 1 November 2026

Choose a dataset and open **Evaluation → The deeper checks**. Bitext and
InsuranceQA include experimental PRDC with neighborhood sensitivity; the three
insurance datasets include expandable individual AI relevance/coherence reviews.
MIMIC includes SDMetrics, missing-lab coverage and the exploratory diagnosis/version
failure screen. InsuranceQA includes two targeted official-source factual
counterexamples. Results are loaded from saved artifacts without running a model.

Pass in the AI review does not mean factual accuracy; no human calibration or
clinical certification is claimed. Clinical structural validity is distinct from
the57/128 diagnosis/version pairs flagged by the narrow new screen. Method and
reproduction details: `reports/advanced_analysis/v1/README.md`.

## Multi-dataset update

Use the **Dataset & model** selector for Bitext, InsuranceQA V2, Insurance
Reliability or MIMIC-IV Demo. Dataset previews, saved outputs and evaluation
results follow the selected source. Each has its own frozen adapter. Models
load on demand; switching datasets unloads the previous model before loading
the next, so only one occupies GPU memory.

The new assessment queue compares128 unfiltered generation attempts from each
original/fine-tuned model, then scores the reserved test split with no updates.
During the queue, live prompting is blocked to reserve the GPU. Source browsing
and saved examples remain available. Pending results are labelled; no training
record is substituted for an unavailable generated output. The evaluation page
refreshes while open and reads completed results automatically.

Open **Terminal → Run Task → Watch new model assessments** in VS Code to monitor
the existing queue. Closing that terminal does not stop its background worker.
Status is `reports/remaining_evaluation_status.json`. After completion, use the
playground to generate live with any of the selected adapters.

Record formats: Bitext has customer/support pairs and labels; InsuranceQA has
domain/question/answer; Reliability has a scenario, context and expected workflow;
MIMIC has structured admission demographics, diagnosis and numeric lab fields.
Reliability expected labels are not evidence that an agent executed the workflow.
MIMIC uses the official100-patient demo, not the full database. Clinical references
use equally weighted field similarities; text references use TF–IDF wording.
Neither is a correctness or causal-attribution score.

The earlier instructions below describe the original Bitext-only view. Its
500-attempt comparison and MAUVE results remain unchanged. New datasets use
their own structural checks, category TV, lengths, copying and held-out loss;
MIMIC also reports numeric ranges, missingness, KS/Wasserstein and correlations.
No new MAUVE or semantic embedding score is claimed for these three datasets.

Local frontend for the selected Gemma 3 4B insurance-data generator. Inference
uses the actual frozen full-pass QLoRA adapter on this computer's NVIDIA GPU.
It does not train or modify the model.

## Start on Windows

From a PowerShell terminal in `D:\gemma-finetune`:

```powershell
.\scripts\run_demo_windows.ps1
```

Open **http://127.0.0.1:8765**. Leave the terminal running; Ctrl+C stops the
server. Only start one server on this port. The model loads when you first
press **Generate a record** or **Load model**. The first load can take a few
minutes; later requests reuse the loaded model. Do not run another GPU
training or generation process at the same time.

In VS Code, use **Terminal → Run Task → Run Simile demo**. The launcher reports
the URL and returns if the demo is already running in another process.

## What to show in the assessment

1. **Playground:** type `Generate synthetic data.`, then generate. The result
   contains a customer request, support reply, intent, category and tags.
   Explore a different topic with the example prompts or optional intent.
   View or download the original record as JSON.
2. **Source perspective:** inspect three nearby training requests and open their
   full records. TF–IDF lexical similarity measures shared wording, not correctness
   or causal training influence. Normalized source matches are shown separately.
3. **Dataset explorer:** search and filter the 31,200 training examples, with
   pagination and full-record previews. These are source examples, not new model
   responses. Bitext itself is a synthetic dataset.
4. **Evaluation:** compare the existing original, pilot and full-pass runs;
   inspect intent coverage, MAUVE, source copying and the final test results.

Saved assessment outputs are explicitly labelled **Saved model output**. A new
response is labelled **Live model output**. Saved sample #12 is the initial
preview; **Next saved example** advances through the unchanged assessment file.
The initial example is a display choice, not an evaluation subset.

## Limits and reproducibility

- This is a synthetic-record generator, not a customer-service chatbot. A prompt
  about coverage asks for a new coverage example rather than answering a real
  customer's policy question. Topic instructions are best effort.
- The default prompt uses the frozen training generation instruction. Evaluation
  used batches of four; this interactive demo generates one record per request.
  Seeds therefore do not guarantee identical assessment outputs. Custom prompts,
  temperatures and intents are exploratory and do not update reported scores.
- Displayed evaluation metrics are loaded from existing artifacts, not recomputed
  for the current response. MAUVE is exploratory, not an accuracy percentage.
- Records retain typos, profanity and placeholders present in the source style.
  Source similarity does not prove insurance correctness or novelty.
- Live jobs are held in server memory. Download a JSON record if you want to keep
  it after stopping the server. The model, dataset and saved assessments stay on D:.
- The server binds only to `127.0.0.1:8765`. No public deployment or remote model API
  is used. Optional Google Fonts fall back to system fonts if unavailable.
- Adapter: `outputs/bitext-4b-full-pass-windows/adapter/`; its SHA256 is verified
  against `outputs/final-assessment/selection.json` before model loading.
- Restart the server after Python changes; reload the browser after HTML/CSS/JS
  changes. The launcher applies the existing D: cache and offline-model settings.

## Files

`server.py` serves assets, dataset/search, recorded results and serialized GPU
generation jobs. `static/` contains the responsive HTML, CSS and JavaScript.
The implementation uses the existing Python ML environment without an extra
frontend build system or classifier training.
