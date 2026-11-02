"""Write a readable report only after actual training and evaluation finish."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="bitext-4b-pilot-windows")
    args = parser.parse_args()
    out = ROOT / "outputs" / args.run
    read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
    evaluation = read(out / "evaluation.json")
    training = read(out / "training_summary.json")
    audit = read(ROOT / "reports/bitext_audit.json")
    source = read(ROOT / "data/bitext/source_manifest.json")
    base, tuned = evaluation["base"], evaluation["tuned"]
    normalized_path = out / "evaluation_unfenced.json"
    normalized = read(normalized_path) if normalized_path.exists() else None

    def number(record, name, percent=False):
        value = record.get(name)
        if value is None:
            return "Not available"
        return f"{100*value:.1f}%" if percent else f"{value:.3f}"

    lines = ["# Bitext Gemma pilot results", "", f"Run: `{args.run}`", "",
             "This report describes a bounded pilot. It does not claim that the assignment's distribution-matching criterion has been met.", "",
             "## Experiment", "",
             f"- Source: [Bitext Insurance]({source['source']}), revision `{source['revision']}`.",
             f"- Source records: {audit['source_rows']:,}; train/validation/test: {audit['split_sizes']}.",
             "- Grouped splits share no normalized requests or responses; semantic paraphrases may remain.",
             f"- Training steps: {training['arguments']['steps']}; peak PyTorch GPU allocation: {training['peak_gpu_allocated_gib']:.3f} GiB.",
             f"- Validation loss: {training['validation_before']['eval_loss']:.4f} before, {training['validation_after']['eval_loss']:.4f} after.",
             "- Base and tuned generations use the same prompt, sampling settings, and seed list. No category quotas or JSON repairs were applied.",
             "- The held-out final test set was not used in this pilot.", "",
             "## Generated records", "", "| Measure | Base | Tuned |", "| --- | ---: | ---: |",
             f"| Attempts | {base['attempted']} | {tuned['attempted']} |",
             f"| Valid records | {base['valid']} | {tuned['valid']} |"]
    for title, key in (("Valid output rate", "valid_rate"), ("Intent coverage", "intent_coverage"),
                       ("Training request matches", "training_request_match_rate"),
                       ("Training response matches", "training_response_match_rate"),
                       ("Exact training record matches", "exact_training_record_match_rate"),
                       ("Duplicate generated records", "duplicate_generated_record_rate")):
        lines.append(f"| {title} | {number(base,key,True)} | {number(tuned,key,True)} |")
    if normalized:
        lines += ["", "## Formatting-controlled comparison", "",
                  "The base model wrapped its outputs in Markdown code fences. Its strict JSON failure rate therefore exaggerates the difference in usable content.",
                  "A separate evaluation removes only a complete outer code fence, applying the same rule to both models. It does not repair JSON fields or labels.",
                  f"After this normalization: base {normalized['base']['valid']}/64 valid; tuned {normalized['tuned']['valid']}/64 valid.",
                  "The distribution comparison below uses these formatting-normalized records; strict results remain in evaluation.json."]
    base_comparison = normalized["base"] if normalized else base
    tuned_comparison = normalized["tuned"] if normalized else tuned
    lines += ["", "## Distribution distances", "",
              "Distances below are conditional on valid records. Inspect invalid-output rates alongside them; excluding failures can make similarity appear better.", "",
              "Lower distances indicate greater similarity in the measured feature, not overall correctness.", "",
              "| Distance from training reference | Base | Tuned |", "| --- | ---: | ---: |"]
    for title, key in (("Intent total variation", "intent_total_variation"), ("Category total variation", "category_total_variation")):
        lines.append(f"| {title} | {number(base_comparison.get('distribution_distances',{}),key)} | {number(tuned_comparison.get('distribution_distances',{}),key)} |")
    for field in ("instruction", "response"):
        for title, key in (("KS statistic", "ks_statistic"), ("Wasserstein distance (characters)", "wasserstein")):
            a = base_comparison.get("distribution_distances",{}).get(f"{field}_length",{})
            b = tuned_comparison.get("distribution_distances",{}).get(f"{field}_length",{})
            lines.append(f"| {field} length: {title} | {number(a,key)} | {number(b,key)} |")
    lines += ["", "## Interpretation limits", "",
              "- A lower validation loss verifies improvement on the language-model loss; it does not prove distribution matching or answer correctness.",
              "- Small generation samples have substantial sampling variation. The JSON report includes real-validation resampling controls for comparison.",
              "- Valid labels and JSON structure do not establish that a response answers its request or that style tags are accurate.",
              "- Manual review forms were exported but have not been reviewed by an insurance expert.",
              "- Copy checks detect exact/normalized matches; they do not rule out memorization or provide a privacy guarantee.",
              "- Bitext is synthetic. Similarity to it does not establish similarity to real customers or the entire U.S. insurance market.",
              "- Embedding similarity and downstream training utility are outside this initial report.", "",
              "## Evidence", "",
              f"- Detailed evaluation: `outputs/{args.run}/evaluation.json`.",
              f"- Training metrics: `outputs/{args.run}/training_summary.json`.",
              f"- Adapter: `outputs/{args.run}/adapter/`.",
              f"- Raw outputs and manual review files: `outputs/{args.run}/`.", ""]
    target = ROOT / "reports/bitext_pilot_report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
