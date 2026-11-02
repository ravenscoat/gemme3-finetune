"""Verify complete frozen batches and report source fidelity without hiding failures."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json

from dataset_catalog import ROOT,CATALOG,read_json,read_rows,sha,value_at,signature
from fidelity_alignment import OUT,ORDER,SIZES,features,markers
from json_io import write_json


def fmt(x):return 'not estimable' if x is None else f'{x:.4g}'


def verify(key):
    d=OUT/key;p=read_json(d/'protocol.json');f=read_json(d/'baseline_freeze.json');done=read_json(d/'completion.json');m=read_json(d/'metrics.json')
    assert sha(d/'protocol.json')==f['protocol_sha256']==done['protocol_sha256']==m['protocol_sha256']
    assert sha(d/'baseline_generations.jsonl')==f['baseline_copy_sha256']
    assert sha(ROOT/p['baseline_path'])==p['baseline_sha256']
    assert sha(ROOT/CATALOG[key]['prepared']/'train.jsonl')==p['source_train_sha256']
    import hashlib
    prompt=(ROOT/CATALOG[key]['prepared']/'generation_prompt.txt').read_text(encoding='utf-8').strip()
    assert hashlib.sha256(prompt.encode()).hexdigest()==p['source_prompt_sha256']
    assert sha(Path(p['adapter'])/'adapter_model.safetensors')==p['adapter_sha256']
    assert sha(d/'controlled_generations.jsonl')==done['raw_sha256']
    assert sha(d/'metrics.json')==done['metrics_sha256']
    for label in ['baseline','controlled']:
        raw=read_rows(d/(label+'_generations.jsonl'));a=m['arms'][label]
        assert len(raw)==p['n'] and [r['sample_id'] for r in raw]==list(range(p['n']))
        assert a['raw_sha256']==sha(d/(label+'_generations.jsonl'))
        assert a['structure_valid']+len(a['failures'])==p['n']
        for row in raw:
            assert row['batch_seed']==p['baseline_batch_seed_start']+row['sample_id']//4
            assert row['batch_size']==4
    extra={}
    parsed=read_rows(d/'controlled_records.jsonl')
    if key=='bitext':
        extra['style_conditions_satisfied']=sum(markers(w['record'])==p['conditions'][w['sample_id']]['markers'] for w in parsed)
    if key=='reliability':
        extra['history_and_action_count_conditions_satisfied']=sum(
            len(w['record']['input']['conversation_history'])==p['conditions'][w['sample_id']]['history_turns'] and
            len(w['record']['expected_output']['required_actions'])==p['conditions'][w['sample_id']]['required_action_count'] for w in parsed)
    return p,m,{'status':'verified','attempts_per_arm':p['n'],'adapter_sha256':p['adapter_sha256'],
        'protocol_sha256':sha(d/'protocol.json'),'source_train_sha256':p['source_train_sha256'],
        'all_raw_attempts_retained':True,'original_baseline_unchanged':True,'adapter_unchanged':True,
        'extra_condition_checks':extra,'elapsed_seconds':done['elapsed_seconds']}


def main():
    status=read_json(ROOT/'reports/fidelity_alignment_status.json')
    assert status['phase']=='complete','Wait until the queue completes; incomplete batches cannot establish the result.'
    lines=['# Training-source fidelity: model-only versus controlled generation','',
        'Development experiment. Each comparison uses the same saved Gemma adapter, with and without externally supplied source profiles. No weights were updated in this experiment.','',
        '**Interpretation:** category quotas, source-supported combinations, numeric/length intervals, style markers and missingness patterns were selected from training data before generation. Improvements partly reflect this controller. They do not establish that the model independently learned the source probabilities. Every generated attempt is included; no repair, reroll or output filtering.','',
        'MIMIC uses the new mixed-prompt candidate in both arms; other datasets use their original selected adapters. Batch size, seed schedule and sampling settings match within each source. Prompts and remaining completion budgets differ. Seed matching is not a paired statistical design across different prompts. No new test scoring or deployment.','',
        'TV measures proportion mismatch (0 is an exact match). Categorical counts and copying checks use the existing normalized signatures (case/punctuation/spacing and numeric representation), with exact source-vocabulary violations reported separately. Copy counts are not bytewise matches or privacy guarantees. Wasserstein measures distribution distance in the field’s original units; lower is closer. Matching ranges alone does not establish matching distributions. All values are descriptive of these fixed small batches; raw KS p-values and Wilson intervals are exploratory and do not prove equivalence or population accuracy.','',
        '| Dataset | Attempts/arm | Valid JSON/schema, before → after | Source constraints, before → after | Primary-label TV, before → after | Whole-record copies, before → after |',
        '|---|---:|---:|---:|---:|---:|']
    bundle={'created_utc':datetime.now(timezone.utc).isoformat(),'datasets':{},'verification':{},'scope':'Fixed small development batches; hybrid source-profile controller plus existing Gemma adapters.'}
    details=[];plot_data={}
    for key in ORDER:
        p,m,v=verify(key);bundle['datasets'][key]=m;bundle['verification'][key]=v
        b,c=m['arms']['baseline'],m['arms']['controlled'];primary=CATALOG[key]['group']
        bv=b['feature_fidelity'][primary]['TV'];cv=c['feature_fidelity'][primary]['TV']
        copies=lambda a:sum(w['exact_training_record_match'] for w in read_rows(OUT/key/(a+'_records.jsonl')))
        lines.append(f'| {CATALOG[key]["label"]} | {p["n"]} | {b["structure_valid"]} → {c["structure_valid"]} | {b["source_constraint_valid"]} → {c["source_constraint_valid"]} | {fmt(bv)} → {fmt(cv)} | {copies("baseline")} → {copies("controlled")} |')
        details += ['',f'## {CATALOG[key]["label"]}','',
            f'Baseline: `{p["baseline_path"]}`. Frozen plan and every raw/parsed record: `{key}/`. Adapter SHA256: `{p["adapter_sha256"]}`.','',
            f'Full-prefix generation budget preflight: `{key}/tokenization_preflight.json`. Token-limit hits: {b["token_limit_count"]} → {c["token_limit_count"]}.',
            f'Controlled exact field requirements: {c["condition_compliance"]["required_values"]}/{p["n"]}; all numeric/length intervals: {c["condition_compliance"]["all_ranges"]}/{p["n"]}; both: {c["condition_compliance"]["both"]}/{p["n"]}. These counters include invalid attempts in their denominator, and exclude style/history/action requirements, which are checked separately below. Numeric endpoint tolerance1e-5 for six-significant-digit rendered bounds.','',
            'Extra supplied-condition checks: '+json.dumps(v['extra_condition_checks'])+f' (denominator {p["n"]}; empty means not applicable).','',
            '| Measured characteristic | TV before | TV after | Change |','|---|---:|---:|---|']
        modes=set(b['feature_fidelity'][primary]['source_counts'])
        coverage=[len(modes&set(a['feature_fidelity'][primary]['generated_counts'])) for a in [b,c]]
        details.insert(len(details)-2,f'Observed source primary-label coverage: {coverage[0]} → {coverage[1]} of {len(modes)} source modes. Mode coverage alone does not measure proportion matching.\n')
        for field,x in b['feature_fidelity'].items():
            old,new=x['TV'],c['feature_fidelity'][field]['TV']
            change='not estimable' if old is None or new is None else 'closer' if new<old-1e-9 else 'farther' if new>old+1e-9 else 'unchanged'
            details.append(f'| {field} | {fmt(old)} | {fmt(new)} | {change} |')
        details += ['', 'Numeric/text-length distributions below use schema-valid rows only. The validity totals above retain all attempts; a low distance does not excuse invalid rows. Source is the complete training split.','',
            '| Field | Wasserstein before → after | KS gap before → after | Source min/median/max | Before min/median/max | After min/median/max | Missing %, source/before/after | Outside source range %, before/after |',
            '|---|---:|---:|---|---|---|---|---|']
        for field,x in b['fidelity_vs_training']['numeric'].items():
            y=c['fidelity_vs_training']['numeric'][field]
            triple=lambda a,label:'/'.join(fmt(a.get(label,{}).get(k)) for k in ['min','median','max'])
            pct=lambda a,k:fmt(None if a.get(k) is None else 100*a[k])
            details.append(f'| {field} | {fmt(x.get("wasserstein"))} → {fmt(y.get("wasserstein"))} | {fmt(x.get("ks_statistic"))} → {fmt(y.get("ks_statistic"))} | {triple(x,"reference")} | {triple(x,"generated")} | {triple(y,"generated")} | {pct(x,"reference_missing_rate")}/{pct(x,"generated_missing_rate")}/{pct(y,"generated_missing_rate")} | {pct(x,"generated_outside_reference_range_rate")}/{pct(y,"generated_outside_reference_range_rate")} |')
        details += ['',f'Exact training request reuse: {fmt(b["training_request_match_rate"])} → {fmt(c["training_request_match_rate"])}; reply reuse: {fmt(b["training_reply_match_rate"])} → {fmt(c["training_reply_match_rate"])} (fractions of valid rows). Whole-record uniqueness: {fmt(b["unique_record_rate"])} → {fmt(c["unique_record_rate"])}. Exact-match checks do not establish privacy.','',
            'Source warning counts before: '+json.dumps(b['source_warning_counts'])+'; after: '+json.dumps(c['source_warning_counts'])+'.',
            'Schema/parse failures before: '+json.dumps(b['failures'])+'; after: '+json.dumps(c['failures'])+'.']
        if key=='mimic_demo':
            details += ['',f'Narrow diagnosis-code/version prefix flags: {b["prefix_version_flags"]} → {c["prefix_version_flags"]}. This is a prefix screen, not a codebook or clinical-validity check. Mean absolute Spearman correlation difference: {fmt(b["fidelity_vs_training"]["spearman_correlations"]["mean_absolute_difference"])} → {fmt(c["fidelity_vs_training"]["spearman_correlations"]["mean_absolute_difference"])}.',
                'Clinical sample: official100-patient MIMIC-IV demo;230 training admissions/80 training patients. The controller supplies most categorical fields, diagnosis codes, decade age bands and numeric deciles/null patterns. This is a strongly controlled cohort generator, not evidence about the full MIMIC database or the U.S. population. No expert clinical review was added.']
        if key=='bitext':details += ['', 'Source itself is synthetic support data. Four planned records per39 intents; measure actual generated counts rather than assuming instruction compliance. Literal style markers approximate selected characteristics; they do not certify every tag or request/reply coherence.']
        if key=='insuranceqa':details += ['', 'Domain/length similarity does not establish factual answer accuracy. Previously identified insurance-fact counterexamples and unblinded AI-review limitations remain relevant.']
        if key=='reliability':details += ['', 'A source-trained benchmark split is an in-domain generation evaluation. Matching route/action-count proportions does not certify the correctness of individual actions or responses.']
        plot_data[key]={'baseline':b,'controlled':c,'primary':primary}
    lines += ['', '**Reading the result:** examine validity, label proportions, continuous distributions, style and copying together. A win in a supplied label field is evidence of pipeline compliance; it is not an independent validation of a learned distribution. Retain all regressions shown below.']+details
    lines += ['', '## Provenance and limits','',
        'Frozen source-only profiles preserve a supported prototype combination while replacing precise numerical values with coarse intervals. They do not provide complete source records to the prompt. The greedy balancing procedure is deterministic and not random sampling; rare modes may be absent at these sample sizes. Profiles, inference settings and hashes are in each protocol. `verification.json` checks source/baseline/model hashes, IDs, seeds, batch sizes, score hashes and attempt accounting.','',
        'No universal pass threshold was supplied by the assignment. These descriptive comparisons cannot prove equality of distributions. Range support, marginal proportions and correlations cover different aspects; full joint/semantic validity needs larger independent confirmation and expert review. Existing final tests have already been used and were not rerun for this experiment.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    originals={}
    for key,spec in CATALOG.items():
        original=read_json(ROOT/spec['selection']);actual=sha(Path(original['adapter'])/'adapter_model.safetensors')
        assert actual==original['adapter_sha256'];originals[key]=actual
    bundle['original_adapter_hashes_verified']=originals
    write_json(OUT/'summary.json',bundle);write_json(OUT/'verification.json',{'datasets':bundle['verification'],'original_adapters_unchanged':originals})
    print('Verified all four batches; report:',OUT/'report.md',flush=True)
    return plot_data


if __name__=='__main__':main()
