"""Persist assistant judgments after inspecting the 20 records and source evidence.

These are explicit qualitative AI judgments, not an automated scorer or human labels.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/joint_review'
NOTES = {
 'R01': 'Travel-policy purchase request, purchase intent, and application procedure agree with source purchase examples.',
 'R02': 'Personal-information change request and update procedure agree with the source change-personal-details intent.',
 'R03': 'Customer-service contact request and contact-number lookup agree. Reply matches a source reply; novelty assessed separately.',
 'R04': 'Monitor health claims request is consistent with track_claim; reply describes viewing claim status.',
 'R05': 'Pet-policy information request and policy-detail lookup agree. Both request wording and reply have close source matches.',
 'R06': 'Pet-policy information request is answered with a pet-policy lookup procedure. Source supports the intent despite informal wording.',
 'R07': 'Life-policy information request and life-policy lookup reply agree with the source intent.',
 'R08': 'Home-policy information request and policy-details lookup agree. Website steps remain template claims, not verified site behavior.',
 'R09': 'Coverage reduction aligns with downgrade_coverage. Repeated downgrade-option steps are clumsy but also appear in source templates; minor clarity issue.',
 'R10': 'Invoice dispute request and dispute-form procedure agree with source dispute_invoice examples.',
 'R11': 'Cancil is interpretable as cancel; cancellation intent and reply agree. The lexically nearest request belongs to comparison, so that neighbor is not treated as ground truth.',
 'R12': 'Home-policy information request and policy-information reply agree with source examples.',
 'R13': 'Health-policy details request and health-policy lookup reply agree with source examples.',
 'R14': 'Travel-policy information request and travel-policy lookup reply agree with source examples.',
 'R15': 'Request for different health policies reasonably supports compare_insurance_policies; reply describes comparison rather than purchase.',
 'R16': 'Auto-policy purchase request and buy_insurance_policy intent agree. Generic INSURANCE_TYPE slots are permitted in this dataset, although using auto-specific wording would be clearer.',
 'R17': 'Home-policy lookup request and home-policy details reply agree with source examples.',
 'R18': 'Payment scheduling request and scheduling procedure agree with source schedule_payments examples.',
 'R19': 'Request about the provider insurance offerings is general_information. A lexically close home-insurance request is not semantically decisive; the general-information source reply supports this interpretation.',
 'R20': 'Imd-payment is unclear and has no literal training-request match. Accepting a payment does not unambiguously mean accepting a settlement; the reply assumes settlement acceptance. Needs clarification.',
}


def main():
    target = OUT / 'ai_judgments.jsonl'
    if target.exists():
        raise RuntimeError('Preserve prior AI judgments; use a new version for re-review')
    rows = [json.loads(line) for line in (OUT / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    keys = {r['review_id']: r for r in map(json.loads, (OUT / 'model_key.jsonl').read_text(encoding='utf-8').splitlines())}
    sources = [json.loads(line) for line in (ROOT/'data/bitext/prepared/train.jsonl').read_text(encoding='utf-8').splitlines()]
    by_line = {r['source_line']: r for r in sources}
    cache, judgments = {}, []
    for row in rows:
        ident = row['review_id']
        key = keys[ident]
        if key['run'] not in cache:
            path = ROOT/'outputs'/key['run']/'tuned_lexical_neighbors.jsonl'
            cache[key['run']] = {r['sample_id']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}
        neighbor = cache[key['run']][key['sample_id']]
        evidence = {}
        for field in ('instruction', 'response'):
            item = neighbor[field]
            source = by_line[item['nearest_source_line']]
            evidence[field] = {'source_line': source['source_line'], 'intent': source['intent'],
                               'instruction': source['instruction'], 'response': source['response'],
                               'cosine_similarity': item['cosine_similarity']}
        judgments.append({'review_id': ident, 'judge_type': 'AI assistant',
                          'verdict': 'Unsure' if ident == 'R20' else 'Pass',
                          'observation': NOTES[ident], 'source_evidence': evidence,
                          'input_record_sha256': hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                          'scope': 'Request-intent-category agreement, response relevance and internal coherence relative to Bitext; tags, novelty and insurance factual correctness not graded'})
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in judgments), encoding='utf-8')
    counts = Counter(r['verdict'] for r in judgments)
    per_model = {model: Counter(r['verdict'] for r in judgments if keys[r['review_id']]['model'] == model)
                 for model in ('pilot', 'full_pass')}
    lines = ['# Source-grounded AI judge review', '',
             'The user explicitly requested the assistant act as judge. This is AI-assisted qualitative review, not human labels or expert certification. The assistant had access to source/model keys while collecting evidence, so this is not claimed as a strictly blinded assessment.', '',
             'Rubric: agreement between request and intent/category, whether the reply addresses the request, and internal coherence relative to Bitext examples. Placeholders, profanity and understandable typos alone do not fail. An unclear request is Unsure rather than an assumed pass. Closeness to source is not novelty. Tags and real-world insurance correctness are not scored.', '',
             f'Observed sample: {counts["Pass"]} Pass, {counts["Fail"]} Fail, {counts["Unsure"]} Unsure out of 20 valid records. These are sample judgments, not model accuracy estimates.', '',
             '| Model | Pass | Fail | Unsure |', '| --- | ---: | ---: | ---: |']
    for model, c in per_model.items():
        lines.append(f'| {model} | {c["Pass"]} | {c["Fail"]} | {c["Unsure"]} |')
    lines += ['', '| Review ID | AI verdict | Reason |', '| --- | --- | --- |']
    lines += [f'| {r["review_id"]} | {r["verdict"]} | {r["observation"]} |' for r in judgments]
    lines += ['', '## Interpretation', '',
              'This easy, small sample does not establish that either model has a higher semantic success rate. It overrepresents policy-information requests and excludes malformed generations. A missing failure here does not erase problems identified in the earlier diagnostic spot check.', '',
              'R20 is a concrete ambiguity. R09 has repeated instructions inherited from source patterns; R16 leaves insurance-type slots generic. R11 and R19 show why lexical-neighbor labels cannot automatically determine semantic correctness.', '',
              'Training data also contains templates and noisy examples. Agreement with it does not establish correct insurance advice or real website workflows. No external policy facts were verified.', '',
              'Recommendation: retain the full-pass adapter as the leading candidate for source fidelity, while reporting its copying rates and all unfiltered results. This review does not independently demonstrate a semantic advantage over the pilot. Do not silently remove failures or copies to improve the reported scores. Final model/settings selection and test assessment remain pending.', '',
              'Evidence and exact judgments: ai_judgments.jsonl. Original human_review fields remain untouched. No final-test records were accessed.', '']
    (OUT/'ai_review_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'overall': dict(counts), 'per_model': {k:dict(v) for k,v in per_model.items()}}, indent=2))


if __name__ == '__main__':
    main()
