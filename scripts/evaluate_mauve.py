"""Exploratory MAUVE with independent GPT-2-large terminal-token features.

Run only after generation/core evaluation. No training, test split, or external
inference service. Equal sample counts and fixed settings for all comparisons.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bitext-4b-evaluation-500-windows'


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='bitext-4b-evaluation-500-windows')
    parser.add_argument('--reference-run', help='Reuse the recorded reference selections, feature model revision, and cached features')
    args = parser.parse_args()
    global OUT
    OUT = ROOT / 'outputs' / args.run
    reference_out = ROOT / 'outputs' / args.reference_run if args.reference_run else None
    fixed = json.loads((reference_out / 'mauve_protocol.json').read_text(encoding='utf-8')) if reference_out else None
    import numpy as np
    import torch
    import faiss
    import mauve
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModel, AutoTokenizer
    faiss.omp_set_num_threads(4)
    protocol = json.loads((OUT / 'generation_protocol.json').read_text(encoding='utf-8'))
    rows = {}
    for label in ('base', 'tuned'):
        raw = load(OUT / f'{label}_generations.jsonl')
        if len(raw) != protocol['samples_per_label']:
            raise RuntimeError('Generation not complete')
        rows[label] = load(OUT / f'{label}_unfenced_manual_review.jsonl')
    n = min(500, *(len(values) for values in rows.values()))
    if fixed:
        n = fixed['samples_per_collection']
        if any(len(values) < n for values in rows.values()):
            raise RuntimeError('Not enough valid outputs for the frozen MAUVE reference; plan a new joint comparison')
    if n < 50:
        raise RuntimeError('Too few valid outputs for this planned comparison')
    rng = random.Random(202610304)
    rows = {label: rng.sample(values, n) for label, values in rows.items()}
    prepared = ROOT / 'data/bitext/prepared'
    rows['training_reference'] = rng.sample(load(prepared / 'train.jsonl'), n)
    rows['validation_control'] = rng.sample(load(prepared / 'validation.jsonl'), n)
    if fixed:
        base_by_id = {row['sample_id']: row for row in load(OUT / 'base_unfenced_manual_review.jsonl')}
        rows['base'] = [base_by_id[i] for i in fixed['sample_ids']['base']]
        for label, split in (('training_reference', 'train'), ('validation_control', 'validation')):
            by_line = {row['source_line']: row for row in load(prepared / f'{split}.jsonl')}
            rows[label] = [by_line[i] for i in fixed['reference_source_lines'][label]]
    model_id = 'openai-community/gpt2-large'
    config_path = OUT / 'mauve_protocol.json'
    if config_path.exists():
        revision = json.loads(config_path.read_text(encoding='utf-8'))['feature_model_revision']
    else:
        revision = fixed['feature_model_revision'] if fixed else HfApi().model_info(model_id).sha
    config = {'feature_model': model_id, 'feature_model_revision': revision,
              'max_tokens': 512, 'batch_size': 4, 'samples_per_collection': n,
              'feature': 'last non-padding token hidden state, float32 GPT-2-large',
              'seeds': [25, 26, 27], 'num_buckets': max(2, round(n / 10)),
              'kmeans_num_redo': 5, 'kmeans_max_iter': 500, 'scaling_factor': 5,
              'sample_ids': {label: [row['sample_id'] for row in rows[label]] for label in ('base', 'tuned')},
              'reference_source_lines': {label: [row['source_line'] for row in rows[label]]
                                         for label in ('training_reference', 'validation_control')},
              'versions': {name: importlib.metadata.version(name) for name in ('mauve-text', 'faiss-cpu', 'torch', 'transformers')}}
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise RuntimeError('MAUVE protocol changed; preserve original results')
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('Downloading/loading pinned GPT-2-large feature model on D:', flush=True)
    model_path = snapshot_download(model_id, revision=revision,
        allow_patterns=['config.json', 'generation_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
                        'merges.txt', 'model.safetensors'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to('cuda').eval()
    cache = OUT / 'mauve_features'
    cache.mkdir(exist_ok=True)
    results = {'protocol': config, 'final_test_used': False,
               'limitations': ['Exploratory: 500 or fewer valid samples; authors recommend a few thousand.',
                               'Conditional on schema-valid outputs; consult core failure counts.',
                               'Not semantic correctness, insurance expertise, or proof of equivalence.',
                               'Seed ranges are sensitivity checks, not confidence intervals.',
                               '512-token truncation may omit text; rates are reported.'],
               'fields': {}}
    for field in ('instruction', 'response'):
        features, truncation = {}, {}
        for label, values in rows.items():
            texts = [row[field] for row in values]
            key = hashlib.sha256(json.dumps([revision, config['max_tokens'], texts], ensure_ascii=False).encode()).hexdigest()
            target = cache / f'{field}_{label}_{key}.npy'
            prior_target = reference_out / 'mauve_features' / target.name if reference_out else None
            lengths = [len(tokenizer.encode(text)) for text in texts]
            truncation[label] = sum(length > config['max_tokens'] for length in lengths) / n
            if target.exists():
                features[label] = np.load(target)
                continue
            if prior_target and prior_target.exists():
                features[label] = np.load(prior_target)
                np.save(target, features[label])
                print(f'Reused evaluator features: {field}/{label}', flush=True)
                continue
            batches = []
            for start in range(0, n, config['batch_size']):
                inputs = tokenizer(texts[start:start + config['batch_size']], return_tensors='pt',
                                   padding=True, truncation=True, max_length=config['max_tokens']).to('cuda')
                with torch.inference_mode():
                    hidden = model(**inputs).last_hidden_state
                    terminal = inputs['attention_mask'].sum(dim=1) - 1
                    batch = hidden[torch.arange(hidden.shape[0], device='cuda'), terminal]
                    batches.append(batch.float().cpu().numpy())
                if start % 100 == 0:
                    print(f'{field}/{label}: {min(start + config["batch_size"], n)}/{n}', flush=True)
            features[label] = np.concatenate(batches)
            np.save(target, features[label])
        scores = {}
        for label in ('base', 'tuned', 'validation_control'):
            scores[label] = []
            for seed in config['seeds']:
                result = mauve.compute_mauve(p_features=features['training_reference'], q_features=features[label],
                    num_buckets=config['num_buckets'], seed=seed, verbose=False,
                    kmeans_num_redo=config['kmeans_num_redo'], kmeans_max_iter=config['kmeans_max_iter'])
                scores[label].append({'seed': seed, 'mauve': float(result.mauve),
                                      'frontier_integral': float(result.frontier_integral)})
        results['fields'][field] = {'scores': scores, 'truncation_rates': truncation}
        (OUT / 'mauve_evaluation.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    lines = ['# Exploratory MAUVE assessment', '',
             f'{n} records per collection; pinned GPT-2-large features, 512-token limit.', '',
             'Higher MAUVE means closer text distributions under this feature model. This is not an accuracy percentage.', '',
             '| Field | Original median (seed range) | Tuned median (seed range) | Real validation median (seed range) |',
             '| --- | ---: | ---: | ---: |']
    for field, value in results['fields'].items():
        cells = []
        for label in ('base', 'tuned', 'validation_control'):
            values = [score['mauve'] for score in value['scores'][label]]
            cells.append(f'{np.median(values):.3f} ({min(values):.3f}–{max(values):.3f})')
        lines.append(f'| {field} | ' + ' | '.join(cells) + ' |')
    lines += ['', *['- ' + limitation for limitation in results['limitations']], '',
              'Exact settings, source selections, truncation rates, and scores: mauve_evaluation.json in the run directory.', '',
              'Metric implementation: https://github.com/krishnap25/mauve', '']
    report_name = 'mauve_500_assessment.md' if args.run == 'bitext-4b-evaluation-500-windows' else f'{args.run}_mauve.md'
    (ROOT / 'reports' / report_name).write_text('\n'.join(lines), encoding='utf-8')
    print('MAUVE assessment complete.', flush=True)


if __name__ == '__main__':
    main()
