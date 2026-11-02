"""Prepare a balanced blinded human-review sample; never manufacture labels."""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/joint_review'


def main():
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):
        raise RuntimeError('Review already exists; preserve selections and annotations')
    rng = random.Random(2026103101)
    selected = []
    for model, run in [('pilot', 'bitext-4b-evaluation-500-windows'),
                       ('full_pass', 'bitext-4b-full-pass-evaluation-500-windows')]:
        path = ROOT / 'outputs' / run / 'tuned_unfenced_manual_review.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        selected.extend((model, run, row) for row in rng.sample(rows, 10))
    rng.shuffle(selected)
    records, key = [], []
    lines = ['# Blind review: 20 generated insurance-support records', '',
             'Read each record before looking at the separate model key. This is a small diagnostic human review, not insurance-expert certification.', '',
             'For each record, write a brief observation, then choose:', '',
             '- **Pass:** the request fits the intent, the reply addresses it, and the record is internally coherent.',
             '- **Fail:** identify a concrete mismatch, contradiction, or unsupported assumption that makes the record unsuitable.',
             '- **Unsure:** the label boundary, insurance procedure, or wording needs clarification or expert judgment.', '',
             'Profanity, typos, and placeholders occur in the source dataset; their presence alone is not a failure. A fluent reply can still miss the request. You are judging coherence and usefulness, not verifying an actual insurer website. Leave uncertain insurance facts unverified.', '',
             'Reply in chat, for example: `R01: Pass — reply addresses the request; R02: Unsure — unclear label.` No fields have been pre-rated.', '',
             'Twenty records total, ten from each model, shuffled. This sample includes only valid outputs; format-failure and copying checks remain separate. Small sample sizes mean substantial uncertainty. Do not infer quality for every intent.', '']
    for i, (model, run, row) in enumerate(selected, 1):
        record = {'review_id': f'R{i:02}',
                  **{field: row[field] for field in ('instruction', 'intent', 'category', 'tags', 'response')},
                  'human_review': {'observation': '', 'verdict': None, 'reviewer': ''}}
        records.append(record)
        key.append({'review_id': record['review_id'], 'model': model, 'run': run, 'sample_id': row['sample_id']})
        lines += [f'## {record["review_id"]}', '',
                  f'**Customer:** {row["instruction"]}', '',
                  f'**Intent:** `{row["intent"]}` · **Category:** `{row["category"]}` · **Tags:** `{row["tags"]}`', '',
                  '**Support reply:**', '', row['response'], '',
                  '**Your observation:**', '', '**Your verdict:** Pass / Fail / Unsure', '', '---', '']
    for name, rows in [('records.jsonl', records), ('model_key.jsonl', key)]:
        (OUT / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    (OUT / 'review.md').write_text('\n'.join(lines), encoding='utf-8')
    (OUT / 'protocol.json').write_text(json.dumps({'seed': 2026103101, 'per_model': 10,
        'sampling': 'uniform random among schema-valid generated records, independently per model',
        'human_labels': 'pending', 'test_used': False}, indent=2) + '\n', encoding='utf-8')
    print(OUT / 'review.md')
    print(json.dumps(records[:2], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
