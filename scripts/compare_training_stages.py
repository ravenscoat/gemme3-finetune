"""Summarize completed development assessments without touching test records."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
pilot_dir = ROOT / 'outputs/bitext-4b-evaluation-500-windows'
full_dir = ROOT / 'outputs/bitext-4b-full-pass-evaluation-500-windows'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    pilot, full = [read(path / 'evaluation_unfenced.json') for path in (pilot_dir, full_dir)]
    pm, fm = [read(path / 'mauve_evaluation.json') for path in (pilot_dir, full_dir)]
    for key in ('samples_per_collection', 'feature_model_revision', 'max_tokens', 'seeds', 'num_buckets', 'reference_source_lines'):
        if pm['protocol'][key] != fm['protocol'][key]:
            raise RuntimeError(f'MAUVE references/settings differ: {key}')
    columns = [pilot['base'], pilot['tuned'], full['tuned']]
    lines = ['# Original, pilot, and full-pass comparison', '',
             'Development assessment with 500 generation attempts per model. The final test split remains unused. Formatting normalization removes complete outer code fences only; failed attempts remain in the denominators.', '',
             '| Measure | Original | Pilot (200 steps) | Full pass (3,900 steps) |',
             '| --- | ---: | ---: | ---: |']
    lines.append('| Valid records | ' + ' | '.join(f'{c["valid"]}/{c["attempted"]}' for c in columns) + ' |')
    lines.append('| Intents represented, of 39 | ' + ' | '.join(str(len(c.get('intent_counts', {}))) for c in columns) + ' |')
    for title, key in [('Intent TV distance (lower is closer)', 'intent_total_variation'), ('Category TV distance (lower is closer)', 'category_total_variation')]:
        lines.append(f'| {title} | ' + ' | '.join(f'{c["distribution_distances"][key]:.3f}' for c in columns) + ' |')
    for title, key in [('Normalized request matches with training', 'training_request_match_rate'), ('Normalized reply matches with training', 'training_response_match_rate'), ('Exact full-record training matches', 'exact_training_record_match_rate')]:
        lines.append(f'| {title} | ' + ' | '.join(f'{c[key]:.1%}' for c in columns) + ' |')
    for field in ('instruction', 'response'):
        values = [statistics.median(score['mauve'] for score in data['fields'][field]['scores'][label])
                  for data, label in ((pm, 'base'), (pm, 'tuned'), (fm, 'tuned'))]
        lines.append(f'| {field} MAUVE median (higher is closer) | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', 'MAUVE uses matched reference samples and equal collection sizes. It is exploratory at this sample size; it is not an accuracy percentage. Distribution metrics are conditional on valid records, and do not establish semantic correctness or equivalence.', '',
              'Human semantic review is pending. Do not choose a final model solely from this table. Inspect the per-intent, style, length, lexical-neighbor and blinded-review artifacts in each run.', '',
              'No C2ST, independent intent-classifier screening, or downstream utility experiment is claimed here.', '']
    target = ROOT / 'reports/training_stage_comparison.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
