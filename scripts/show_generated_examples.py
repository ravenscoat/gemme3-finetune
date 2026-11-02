"""Readable source-versus-generation evidence; no new generation or filtering."""
import html
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def cell(text):
    return html.escape(str(text)).replace('\n', '<br>').replace('|', '&#124;')


def main():
    out = ROOT/'outputs/bitext-4b-full-pass-evaluation-500-windows'
    generated = load(out/'tuned_unfenced_manual_review.jsonl')
    train = load(ROOT/'data/bitext/prepared/train.jsonl')
    by_line = {r['source_line']:r for r in train}
    neighbors = {r['sample_id']:r for r in load(out/'tuned_lexical_neighbors.jsonl')}
    evaluation = json.loads((out/'evaluation_unfenced.json').read_text())['tuned']
    mauve = json.loads((out/'mauve_evaluation.json').read_text())
    distances = evaluation['distribution_distances']
    lines = ['# How close are our generated examples to Bitext?', '',
             'Final selected model: Gemma 3 4B QLoRA, full-pass checkpoint 3,900. These are actual saved outputs, not newly written demonstrations. Bitext itself is synthetic; this comparison measures resemblance to Bitext, not the whole U.S. insurance market.', '',
             '## Scores tied to the assessment requirement', '',
             '| Property | Full-pass result | Interpretation |', '| --- | --- | --- |',
             '| Structure | 500/500 schema-valid records | Every attempt had the expected fields/allowed labels. Meaning is a separate question. |',
             f'| Intent balance | TV {distances["intent_total_variation"]:.3f}; 37/39 intents | Lower TV is closer; missing intents are appeal_denied_insurance_claim and pay. |',
             f'| Category balance | TV {distances["category_total_variation"]:.3f} | Lower is closer. Not a correctness percentage. |']
    for field, name in [('instruction', 'Customer requests'), ('response', 'Support replies')]:
        scores = mauve['fields'][field]['scores']
        a = statistics.median(r['mauve'] for r in scores['tuned'])
        b = statistics.median(r['mauve'] for r in scores['validation_control'])
        lines.append(f'| {name}: MAUVE | {a:.3f}; held-out Bitext comparison {b:.3f} | Higher is closer in GPT-2-large feature space. Not accuracy. |')
        lens = distances[f'{field}_length']
        lines.append(f'| {name}: length distributions | KS {lens["ks_statistic"]:.3f}; Wasserstein {lens["wasserstein"]:.2f} characters | Lower distances mean closer length distributions. |')
    lines += [f'| Normalized request reproduction | {evaluation["training_request_match_rate"]:.1%} | Some requests repeat source wording. |',
              f'| Normalized reply reproduction | {evaluation["training_response_match_rate"]:.1%} | Strong template reproduction; similarity does not establish novelty. |',
              f'| Exact complete-record reproduction | {evaluation["exact_training_record_match_rate"]:.1%} (3/500) | Three complete five-field records exactly match training. |', '',
              'MAUVE uses 483 valid records per collection with matched references and three clustering seeds. This is exploratory; the authors recommend larger samples. Core metrics use all 500 valid outputs. No test-driven changes were made.', '',
              '## Observed length ranges', '',
              '| Field | Training min / median / max | Generated min / median / max |', '| --- | --- | --- |']
    for field in ('instruction', 'response'):
        values = [[len(r[field]) for r in group] for group in (train, generated)]
        lines.append(f'| {field} (characters) | ' + ' | '.join(f'{min(v)} / {statistics.median(v):g} / {max(v)}' for v in values) + ' |')
    lines += ['', 'Matching typical lengths does not ensure matching rare extremes. Range limits depend strongly on sample size: training contains 31,200 records versus 500 generated records.', '',
              '## Actual outputs alongside training examples', '',
              'The first six records below are sampled uniformly with seed 42. The final example (ID 88) is deliberately included to show a confirmed complete copy. This display is illustrative; the aggregate scores above use the full unfiltered collection.', '',
              'Each source record is selected as the closest reply by word unigram/bigram TF-IDF cosine. It is not a paired gold answer or proof of semantic correctness. Source IDs identify CSV source lines, not positions in the prepared split.', '']
    selected = random.Random(42).sample(generated, 6)
    selected.append(next(r for r in generated if r['sample_id'] == 88))
    for row in selected:
        neighbor = neighbors[row['sample_id']]['response']
        source = by_line[neighbor['nearest_source_line']]
        lines += [f'### Generated ID {row["sample_id"]}', '',
                  '| | Generated record | Closest training reply record |', '| --- | --- | --- |',
                  f'| Source ID | Generation {row["sample_id"]} | CSV line {source["source_line"]} |']
        for field in ('intent', 'category', 'instruction', 'response', 'tags'):
            lines.append(f'| {field} | {cell(row[field])} | {cell(source[field])} |')
        lines += ['', f'Reply lexical similarity: {neighbor["cosine_similarity"]:.3f}. Request seen in training after normalization: {row["request_seen_in_training"]}. Reply seen after normalization: {row["response_seen_in_training"]}.', '']
    lines += ['## What we can conclude', '',
              'The model reproduces Bitext structure, typical lengths and reply style well under these measurements. Intent/category frequencies still differ, two intents are absent in the generated collection, and copying is material. There is no single supported overall similarity percentage.', '',
              'A concrete semantic limitation is generated ID 327: the request asks for information about payment methods, but the reply specifically directs a debit-card transaction. Source-like wording can coexist with an incomplete or overly specific reply.', '',
              'The separate source-grounded AI judge review assessed 20 records: 19 coherence passes and one unclear request. This small AI-reviewed sample does not establish insurance correctness, precise semantic accuracy, or a semantic advantage over the pilot.', '',
              'Reserved test assessment covered 3,900 records after selection: mean completion loss 0.2803. It is supplementary held-out prediction evidence, not a synthetic-response correctness score.', '',
              'Evidence: training_stage_comparison.md, final_test_assessment.md, joint_review/ai_review_report.md, and the JSON/raw-generation artifacts in the run directory.', '']
    target = ROOT/'reports/generated_examples_comparison.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
