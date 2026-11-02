"""Add lexical-neighbor, style, diversity, and blinded-review artifacts.

These are descriptive development checks, not semantic correctness scores.
No test records or external model/API are used.
"""
import argparse
import json
from pathlib import Path
import random
import re

from evaluate_bitext import normalize, numeric_distance

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def style(rows):
    def rate(check):
        return sum(check(row) for row in rows) / len(rows) if rows else None
    return {
        'request_contains_profanity_marker': rate(lambda r: bool(re.search(r'\b(?:fuck\w*|shit\w*|damn)\b', r['instruction'], re.I))),
        'request_contains_negation_marker': rate(lambda r: bool(re.search(r"\b(?:not|never|no|don['’]t|can['’]t|won['’]t)\b", r['instruction'], re.I))),
        'request_ends_question_mark': rate(lambda r: r['instruction'].rstrip().endswith('?')),
        'request_starts_lowercase': rate(lambda r: r['instruction'].lstrip()[:1].islower()),
        'response_contains_placeholder': rate(lambda r: bool(re.search(r'\{\{[^{}]+\}\}', r['response']))),
    }


def diversity(rows):
    n = len(rows)
    return {field: {'unique_normalized': len({normalize(r[field]) for r in rows}),
                    'repetition_rate': 1 - len({normalize(r[field]) for r in rows}) / n if n else None}
            for field in ('instruction', 'response')}


def per_intent(rows, train):
    result = {}
    for intent in sorted({row['intent'] for row in train}):
        reference = [row for row in train if row['intent'] == intent]
        generated = [row for row in rows if row['intent'] == intent]
        entry = {'valid_count': len(generated),
                 'share_of_valid_outputs': len(generated) / len(rows) if rows else None,
                 'training_share': len(reference) / len(train),
                 'missing': not generated,
                 'small_sample_warning': len(generated) < 20}
        if generated:
            entry['diversity'] = diversity(generated)
            entry['length_distances'] = {
                field: numeric_distance([len(row[field]) for row in generated],
                                        [len(row[field]) for row in reference])
                for field in ('instruction', 'response')}
        result[intent] = entry
    return result


def nearest(reference, queries):
    """Return exact top cosine neighbors for sparse, nonnegative TF-IDF vectors."""
    import numpy as np
    scores, indices = [], []
    for start in range(0, queries.shape[0], 64):
        similarity = queries[start:start + 64] @ reference.T
        scores.extend(similarity.max(axis=1).toarray().ravel().tolist())
        indices.extend(np.asarray(similarity.argmax(axis=1)).ravel().tolist())
    return scores, indices


def score_summary(values):
    import numpy as np
    return {'median': float(np.median(values)), 'p90': float(np.quantile(values, .9)),
            'fraction_at_least_0_9': float(np.mean(np.asarray(values) >= .9))}


def write_blinded_review(out, records, count=50):
    # Random sample of valid rows, not a semantic pass-rate estimate for all attempts.
    # Keep the key local and separate from the review form.
    rng = random.Random(202610301)
    selected = []
    for label, rows in records.items():
        selected.extend((label, row) for row in rng.sample(rows, min(count, len(rows))))
    rng.shuffle(selected)
    reviews, key = [], []
    for i, (label, row) in enumerate(selected):
        review_id = f'R{i + 1:03}'
        reviews.append({'review_id': review_id,
                        **{k: row[k] for k in ('instruction', 'response', 'intent', 'category', 'tags')},
                        'human_review': {'observations': '', 'request_intent_agree': None,
                                         'response_addresses_request': None,
                                         'internally_consistent': None,
                                         'overall_usable': None, 'reviewer': ''}})
        key.append({'review_id': review_id, 'model_label': label, 'sample_id': row['sample_id']})
    for name, rows in [('blinded_review.jsonl', reviews), ('blinded_review_key.jsonl', key)]:
        path = out / name
        if path.exists():
            # Never overwrite human annotations on rerun.
            continue
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='bitext-4b-evaluation-1000-windows')
    args = parser.parse_args()
    out = ROOT / 'outputs' / args.run
    protocol_path = out / 'generation_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else None
    raw = {label: load(out / f'{label}_generations.jsonl') for label in ('base', 'tuned')}
    if protocol and any(len(rows) != protocol['samples_per_label'] for rows in raw.values()):
        raise RuntimeError('Incomplete generation: do not write a final evaluation report yet')
    evaluation = json.loads((out / 'evaluation_unfenced.json').read_text(encoding='utf-8'))
    strict = json.loads((out / 'evaluation.json').read_text(encoding='utf-8'))
    records = {label: load(out / f'{label}_unfenced_manual_review.jsonl') for label in raw}
    for label in raw:
        if evaluation[label]['attempted'] != len(raw[label]) or evaluation[label]['valid'] != len(records[label]):
            raise RuntimeError('Evaluation is stale; rerun strict and unfenced evaluators')
    prepared = ROOT / 'data/bitext/prepared'
    train, validation = [load(prepared / f'{split}.jsonl') for split in ('train', 'validation')]
    rng = random.Random(202610302)
    control = rng.sample(validation, min(max(len(r) for r in records.values()), len(validation)))
    report = {
        'run': args.run, 'protocol': protocol, 'final_test_used': False,
        'style_marker_definitions': 'Literal regex/character checks, not language understanding or tag validation.',
        'style': {label: style(rows) for label, rows in {**records, 'training_reference': train, 'validation_control': control}.items()},
        'diversity': {label: diversity(rows) for label, rows in {**records, 'validation_control': control}.items()},
        'lexical_similarity_method': 'Training-fitted word unigram/bigram TF-IDF cosine; max_features=50000; not embeddings, attribution, or proof of copying.',
        'near_copy_flag_threshold': .9, 'threshold_status': 'Exploratory review flag, not a validated plagiarism threshold.',
        'per_intent': {label: per_intent(rows, train) for label, rows in records.items()},
        'nearest_training_similarity': {}, 'semantic_review_status': 'Human labels pending; no semantic pass rate claimed.',
    }
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    neighbors = {label: [{'sample_id': row['sample_id']} for row in rows] for label, rows in records.items()}
    for field in ('instruction', 'response'):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, dtype=np.float32,
                                     strip_accents='unicode', token_pattern=r'(?u)\b\w+\b')
        reference = vectorizer.fit_transform([r[field] for r in train])
        summaries = {}
        for label, rows in {**records, 'validation_control': control}.items():
            if not rows:
                summaries[label] = None
                continue
            scores, indices = nearest(reference, vectorizer.transform([r[field] for r in rows]))
            summaries[label] = score_summary(scores)
            if label in neighbors:
                for result, score, index in zip(neighbors[label], scores, indices):
                    result[field] = {'cosine_similarity': score, 'nearest_source_line': train[index]['source_line'],
                                     'nearest_group_id': train[index]['group_id'], 'nearest_training_text': train[index][field]}
        report['nearest_training_similarity'][field] = summaries
        print(f'Finished lexical comparison: {field}', flush=True)
    for label, rows in neighbors.items():
        (out / f'{label}_lexical_neighbors.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    write_blinded_review(out, records)
    (out / 'extended_evaluation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Bitext generation quality assessment', '', f'Run: `{args.run}`', '',
             'Development evaluation of the saved 200-step adapter versus its frozen base. No additional fine-tuning or final-test evaluation.', '',
             '## Format and coverage', '', '| Measure | Base | Tuned |', '| --- | ---: | ---: |']
    for title, a, b in [
        ('Attempts', len(raw['base']), len(raw['tuned'])),
        ('Strict valid JSON/schema', strict['base']['valid'], strict['tuned']['valid']),
        ('Valid after outer-fence removal', evaluation['base']['valid'], evaluation['tuned']['valid']),
        ('Intents covered (of 39)', len(evaluation['base'].get('intent_counts', {})), len(evaluation['tuned'].get('intent_counts', {})))]:
        lines.append(f'| {title} | {a} | {b} |')
    lines += ['', 'Both models use the same prompt and sampling protocol. No quota balancing, JSON repairs, rejection sampling, or removal of failed attempts. Distribution metrics below are conditional on schema-valid, outer-fence-normalized records.', '',
              '## Distribution and copying', '', '| Measure | Base | Tuned |', '| --- | ---: | ---: |']
    for title, key in [('Intent TV distance (lower is closer)', 'intent_total_variation'), ('Category TV distance (lower is closer)', 'category_total_variation')]:
        values = [evaluation[label].get('distribution_distances', {}).get(key) for label in records]
        lines.append(f'| {title} | ' + ' | '.join(f'{v:.3f}' if v is not None else 'N/A' for v in values) + ' |')
    for title, key in [('Normalized training request matches', 'training_request_match_rate'), ('Normalized training response matches', 'training_response_match_rate'), ('Exact full-record training matches', 'exact_training_record_match_rate')]:
        values = [evaluation[label].get(key) for label in records]
        lines.append(f'| {title} | ' + ' | '.join(f'{v:.1%}' if v is not None else 'N/A' for v in values) + ' |')
    lines += ['', '## Length distributions', '',
              '| Field / distance from training | Base | Tuned |', '| --- | ---: | ---: |']
    for field in ('instruction', 'response'):
        for metric in ('ks_statistic', 'wasserstein'):
            values = [evaluation[label].get('distribution_distances', {}).get(f'{field}_length', {}).get(metric)
                      for label in records]
            lines.append(f'| {field}: {metric} | ' + ' | '.join(f'{v:.3f}' if v is not None else 'N/A' for v in values) + ' |')
    lines += ['', 'Wasserstein distances are in characters. KS distances summarize separation between length distributions. Lower is closer for these features; neither metric checks meaning.', '',
              '## Sampling calibration', '',
              'The core evaluator compares 20 seeded, size-matched subsets of real validation data against training. These observed ranges provide context for finite-sample variation; they are not confidence intervals or equivalence tests.', '',
              '| Model / distance | Generated | Real-validation min | Median | Max |', '| --- | ---: | ---: | ---: | ---: |']
    for label in records:
        for key in ('intent_total_variation', 'category_total_variation'):
            value = evaluation[label].get('distribution_distances', {}).get(key)
            calibration = evaluation[label].get('real_validation_calibration', {}).get(key)
            if calibration:
                lines.append(f'| {label}: {key} | {value:.3f} | {calibration["min"]:.3f} | {calibration["median"]:.3f} | {calibration["max"]:.3f} |')
    lines += ['', '## Language characteristics', '', 'Literal marker frequencies; these do not determine whether an example is correct.', '',
              '| Marker | Training | Base | Tuned |', '| --- | ---: | ---: | ---: |']
    for name in report['style']['training_reference']:
        values = [report['style'][label][name] for label in ('training_reference', 'base', 'tuned')]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.1%}' if v is not None else 'N/A' for v in values) + ' |')
    lines += ['', '## Language tag frequencies', '',
              'Presence of each source tag character per valid record. Matching tag frequencies does not establish that tags accurately describe the text.', '',
              '| Tag | Training | Base | Tuned |', '| --- | ---: | ---: | ---: |']
    tag_chars = sorted({tag for row in train for tag in row['tags']})
    for tag in tag_chars:
        values = [sum(tag in r['tags'] for r in rows) / len(rows) if rows else None
                  for rows in (train, records['base'], records['tuned'])]
        lines.append(f'| {tag} | ' + ' | '.join(f'{v:.1%}' if v is not None else 'N/A' for v in values) + ' |')
    lines += ['', '## Lexical similarity to training', '',
              'TF-IDF cosine measures shared wording. A high score can reflect templates and common insurance vocabulary. It is not proof of memorization. Held-out validation examples are included as a reference.', '',
              '| Field / statistic | Base | Tuned | Real validation control |', '| --- | ---: | ---: | ---: |']
    for field, summaries in report['nearest_training_similarity'].items():
        for metric in ('median', 'fraction_at_least_0_9'):
            values = [summaries.get(label) for label in ('base', 'tuned', 'validation_control')]
            lines.append(f'| {field}: {metric} | ' + ' | '.join(f'{v[metric]:.3f}' if v else 'N/A' for v in values) + ' |')
    lines += ['', '## Results per intent', '',
              'Counts use valid outputs. Missing intents are explicit. Fewer than 20 records is a descriptive small-sample flag, not a statistical acceptance threshold. Detailed conditional length and diversity checks are in extended_evaluation.json.', '',
              '| Intent | Base valid count | Tuned valid count |', '| --- | ---: | ---: |']
    for intent in report['per_intent']['base']:
        lines.append(f'| {intent} | {report["per_intent"]["base"][intent]["valid_count"]} | {report["per_intent"]["tuned"][intent]["valid_count"]} |')
    lines += ['', '## Semantic review', '',
              'Pending. `blinded_review.jsonl` contains up to 50 randomly selected valid records per model, shuffled with model labels in a separate key. Human fields remain empty. This is not an expert correctness assessment.', '',
              'Review whether request and intent agree, the response addresses the request, and the record is internally consistent. Write observations before assigning overall usability. Style tags require a separate interpretation of the source definitions.', '',
              '## Decision limits', '',
              '- No global pass/fail threshold has been established. Inspect all measured dimensions and the real-validation calibration in evaluation_unfenced.json.',
              '- Similar distributions and valid labels do not prove correct insurance advice.',
              '- Matching Bitext does not establish matching real customers or the U.S. market.',
              '- Gradient alignment, data attribution, embeddings, and downstream utility are outside this phase.',
              '- Do not change prompts or select a longer training run based only on lower validation loss.', '',
              'Detailed evidence: evaluation.json, evaluation_unfenced.json, extended_evaluation.json, and raw generations in this run directory.', '']
    target = ROOT / 'reports' / f'{args.run}_assessment.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(target, flush=True)


if __name__ == '__main__':
    main()
