"""Evaluate raw generations without silently repairing or dropping failures."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def strip_outer_json_fence(text):
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text.strip(), flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def tv(a, b):
    ca, cb = Counter(a), Counter(b)
    return .5 * sum(abs(ca[k]/len(a)-cb[k]/len(b)) for k in ca.keys() | cb.keys())


def numeric_distance(a, b):
    a, b = sorted(a), sorted(b)
    points = sorted(set(a+b))
    i = j = 0
    ks = wasserstein = 0.
    previous, previous_gap = points[0], 0.
    for point in points:
        wasserstein += (point-previous)*previous_gap
        while i < len(a) and a[i] <= point:
            i += 1
        while j < len(b) and b[j] <= point:
            j += 1
        previous_gap = abs(i/len(a)-j/len(b))
        ks = max(ks, previous_gap)
        previous = point
    return {"ks_statistic": ks, "wasserstein": wasserstein}


def distributions(records, reference):
    result = {}
    for field in ("intent", "category"):
        result[f"{field}_total_variation"] = tv([r[field] for r in records], [r[field] for r in reference])
    for field in ("instruction", "response"):
        lengths = [len(r[field]) for r in records]
        result[f"{field}_length"] = {**numeric_distance(lengths, [len(r[field]) for r in reference]),
                                     "generated_min": min(lengths), "generated_max": max(lengths),
                                     "generated_median": statistics.median(lengths)}
    for tag in "BCDEGIKLMNPQRWYZ":
        result[f"tag_{tag}_frequency_gap"] = abs(
            sum(tag in r["tags"] for r in records)/len(records) -
            sum(tag in r["tags"] for r in reference)/len(reference))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="bitext-4b-pilot")
    parser.add_argument("--strip-code-fence", action="store_true", help="Secondary comparison: remove only a complete outer JSON code fence")
    args = parser.parse_args()
    prepared = ROOT / "data/bitext/prepared"
    load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    train, validation = [load(prepared / f"{s}.jsonl") for s in ("train", "validation")]
    schema = json.loads((prepared / "schema.json").read_text())
    fields = schema["fields"]
    seen_q = {normalize(r["instruction"]) for r in train}
    seen_a = {normalize(r["response"]) for r in train}
    seen_rows = {tuple(r[f] for f in fields) for r in train}
    out = ROOT / "outputs" / args.run
    report = {"run": args.run, "comparison": "Training reference and validation calibration; final test remains unused",
              "format_normalization": "outer code fence only" if args.strip_code_fence else "none (strict raw output)",
              "limitations": ["Label validity does not establish semantic correctness.",
                              "Small generation samples cannot establish all-intent coverage or equivalence.",
                              "Copy checks are exact/normalized matches, not a privacy guarantee.",
                              "No embedding metric is used in this initial report."]}
    for label in ("base", "tuned"):
        path = out / f"{label}_generations.jsonl"
        if not path.exists():
            continue
        raw = load(path)
        valid, failures, review = [], Counter(), []
        fences_removed = 0
        for item in raw:
            try:
                text = item["text"].strip()
                if args.strip_code_fence:
                    normalized_text = strip_outer_json_fence(text)
                    fences_removed += normalized_text != text
                    text = normalized_text
                record = json.loads(text)
                if not isinstance(record, dict) or set(record) != set(fields):
                    raise ValueError("schema_keys")
                if any(not isinstance(record[f], str) or not record[f].strip() for f in fields):
                    raise ValueError("field_types_or_empty")
                if schema["intent_category"].get(record["intent"]) != record["category"]:
                    raise ValueError("intent_category")
                if any(c not in schema["tag_characters"] for c in record["tags"]):
                    raise ValueError("tag_alphabet")
            except json.JSONDecodeError:
                failures["invalid_json"] += 1
                continue
            except ValueError as error:
                failures[str(error)] += 1
                continue
            valid.append(record)
            review.append({"sample_id": item["sample_id"], **record,
                           "request_seen_in_training": normalize(record["instruction"]) in seen_q,
                           "response_seen_in_training": normalize(record["response"]) in seen_a,
                           "review": {"request_intent_agree": None, "response_addresses_request": None,
                                      "tags_match_style": None, "notes": ""}})
        result = {"attempted": len(raw), "valid": len(valid), "valid_rate": len(valid)/max(1,len(raw)),
                  "failures": dict(failures), "outer_code_fences_removed": fences_removed,
                  "token_limit_hits": sum(r["hit_token_limit"] for r in raw)}
        if valid:
            result.update({"intent_coverage": len(set(r["intent"] for r in valid))/len(schema["intent_category"]),
                           "intent_counts": dict(Counter(r["intent"] for r in valid)),
                           "training_request_match_rate": sum(normalize(r["instruction"]) in seen_q for r in valid)/len(valid),
                           "training_response_match_rate": sum(normalize(r["response"]) in seen_a for r in valid)/len(valid),
                           "exact_training_record_match_rate": sum(tuple(r[f] for f in fields) in seen_rows for r in valid)/len(valid),
                           "duplicate_generated_record_rate": 1-len(set(tuple(r[f] for f in fields) for r in valid))/len(valid),
                           "distribution_distances": distributions(valid, train)})
            rng = random.Random(20261029)
            controls = [distributions(rng.sample(validation, min(len(valid),len(validation))), train) for _ in range(20)]
            result["real_validation_calibration"] = {key: {"min": min(c[key] for c in controls),
                                                          "median": statistics.median(c[key] for c in controls),
                                                          "max": max(c[key] for c in controls)}
                                                     for key in ("intent_total_variation", "category_total_variation")}
            for field in ("instruction_length", "response_length"):
                result["real_validation_calibration"][field] = {
                    metric: {"min": min(c[field][metric] for c in controls),
                             "median": statistics.median(c[field][metric] for c in controls),
                             "max": max(c[field][metric] for c in controls)}
                    for metric in ("ks_statistic", "wasserstein")}
        suffix = "_unfenced" if args.strip_code_fence else ""
        with (out / f"{label}{suffix}_manual_review.jsonl").open("w", encoding="utf-8") as stream:
            for row in review:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        report[label] = result
    filename = "evaluation_unfenced.json" if args.strip_code_fence else "evaluation.json"
    (out / filename).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
