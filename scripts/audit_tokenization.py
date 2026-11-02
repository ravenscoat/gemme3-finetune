"""Check complete JSON targets fit the training context before loading weights."""
import json
import os
from pathlib import Path
import statistics

os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/bitext/prepared"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / ".cache/models/gemma3-4b-training", local_files_only=True)
    prompt = (source / "generation_prompt.txt").read_text(encoding="utf-8").strip()
    fields = json.loads((source / "schema.json").read_text(encoding="utf-8"))["fields"]
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    rows = [json.loads(line) for line in (source / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": json.dumps({f: r[f] for f in fields}, ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=False) for r in rows]
    lengths = []
    prefix_failures = 0
    for start in range(0, len(texts), 256):
        batch = tokenizer(texts[start:start+256], add_special_tokens=False)["input_ids"]
        lengths.extend(map(len, batch))
        prefix_failures += sum(ids[:len(prefix_ids)] != prefix_ids for ids in batch)
    result = {"training_rows": len(rows), "prompt_tokens": len(prefix_ids),
              "minimum": min(lengths), "maximum": max(lengths), "median": statistics.median(lengths),
              "over_1024": sum(n > 1024 for n in lengths),
              "over_1536": sum(n > 1536 for n in lengths),
              "response_mask_prefix_failures": prefix_failures}
    (ROOT / "reports/tokenization_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
    if prefix_failures:
        raise ValueError("Prompt prefix mismatch would invalidate response loss masking")


if __name__ == "__main__":
    main()
