"""Audit source data and make deterministic, group-separated generation splits."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/bitext/prepared"
FIELDS = ("intent", "category", "tags", "instruction", "response")
SEED = 20261029


def normalized(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    source = next((ROOT / "data/bitext/raw").glob("*.csv"))
    with source.open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    if set(original[0]) != set(FIELDS):
        raise ValueError("Unexpected source schema")
    rows, seen, duplicates = [], set(), 0
    for line, row in enumerate(original, 2):
        if any(not isinstance(row[field], str) or not row[field].strip() for field in FIELDS):
            raise ValueError(f"Missing field at CSV line {line}")
        signature = tuple(row[field] for field in FIELDS)
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
        rows.append({**row, "source_line": line})
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        parent[find(i)] = find(j)

    # Group shared normalized questions, word-order variants, and shared answers.
    # This is a conservative heuristic, not a guarantee of semantic independence.
    signatures = {}
    for i, row in enumerate(rows):
        question = normalized(row["instruction"])
        for key in (("question", question),
                    ("question_words", " ".join(sorted(question.split()))),
                    ("answer", normalized(row["response"]))):
            if key in signatures:
                join(i, signatures[key])
            else:
                signatures[key] = i
    groups = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    # Keep large connected components in training rather than contaminating splits.
    # Within remaining groups, balance total row counts per intent approximately.
    rng = random.Random(SEED)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    group_list.sort(key=len, reverse=True)
    intent_totals = collections.Counter(row["intent"] for row in rows)
    split_rows = {split: [] for split in ("train", "validation", "test")}
    counts = {split: collections.Counter() for split in split_rows}
    proportions = {"train": .8, "validation": .1, "test": .1}
    for indices in group_list:
        composition = collections.Counter(rows[i]["intent"] for i in indices)
        # Choose the split with the largest proportional remaining demand.
        scores = {split: sum(n * (intent_totals[intent] * proportions[split] - counts[split][intent])
                            for intent, n in composition.items())
                  for split in split_rows}
        selected = max(scores, key=scores.get)
        group_id = hashlib.sha256("\n".join(sorted(str(rows[i]["source_line"]) for i in indices)).encode()).hexdigest()[:16]
        for i in indices:
            split_rows[selected].append({**rows[i], "group_id": group_id})
        counts[selected].update(composition)
    # Fail rather than silently publishing overlapping questions/answers/groups.
    for field in ("instruction", "response", "group_id"):
        sets = {split: {normalized(r[field]) for r in values} for split, values in split_rows.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert not sets[a] & sets[b], (field, a, b)
    for split in split_rows:
        rng.shuffle(split_rows[split])
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in split_rows[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    mapping = collections.defaultdict(set)
    for row in split_rows["train"]:
        mapping[row["intent"]].add(row["category"])
    assert all(len(categories) == 1 for categories in mapping.values())
    mapping = {intent: next(iter(categories)) for intent, categories in sorted(mapping.items())}
    tag_chars = sorted(set("".join(r["tags"] for r in split_rows["train"])))
    prompt = (
        "Generate synthetic data.\n"
        "Create one new Bitext-style insurance customer-support record. Return only a JSON object "
        "with string fields intent, category, tags, instruction, response. instruction is the "
        "customer request and response is the matching support reply. Keep template placeholders "
        "in double braces where appropriate. Choose the intent naturally; no category quotas are imposed. "
        "Use a consistent intent/category pair from this mapping: " + json.dumps(mapping) +
        ". tags describes the request's language variations using characters from " + "".join(tag_chars) + "."
    )
    (OUT / "generation_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (OUT / "schema.json").write_text(json.dumps({"fields": list(FIELDS), "intent_category": mapping,
                                                "tag_characters": tag_chars}, indent=2) + "\n", encoding="utf-8")
    audit = {
        "source_rows": len(original), "exact_duplicate_rows_removed": duplicates,
        "retained_rows": len(rows), "intents": len(mapping),
        "categories": len(set(mapping.values())), "seed": SEED,
        "grouping": "Connected components of normalized request, sorted request words, or normalized response equality",
        "group_count": len(groups), "largest_group": max(map(len, groups.values())),
        "split_sizes": {s: len(v) for s, v in split_rows.items()},
        "split_intents": {s: dict(c) for s, c in counts.items()},
        "zero_cross_split_normalized_question_response_group_overlap": True,
        "caveat": "Semantic paraphrases may remain across splits; grouping is heuristic. Split proportions are approximate.",
        "source_intents": dict(intent_totals),
        "training_length_chars": {field: {"min": min(len(r[field]) for r in split_rows["train"]),
                                          "max": max(len(r[field]) for r in split_rows["train"]),
                                          "median": statistics.median(len(r[field]) for r in split_rows["train"])}
                                  for field in ("instruction", "response")},
    }
    (reports / "bitext_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("source_rows", "intents", "categories", "group_count", "largest_group", "split_sizes")}, indent=2))


if __name__ == "__main__":
    main()
