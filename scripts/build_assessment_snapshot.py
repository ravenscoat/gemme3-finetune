"""Consolidate selected development evidence without replacing live model defaults."""
import json
from pathlib import Path

from dataset_catalog import ROOT,CATALOG,data,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'reports/assessment_snapshot'
PREFIX=ROOT/'outputs/prefix-alignment-v1'
CLINICAL=ROOT/'outputs/mimic-demo-fidelity-conditioned-v1-windows'


def make_item(key,chosen,baseline,policy,records,metrics_file,notes,baseline_label,adapter):
    rows=read_rows(records);assert len(rows)==chosen['structure_valid']
    assert len({r['sample_id'] for r in rows})==len(rows)
    assert chosen['structure_valid']+len(chosen['failures'])==chosen['attempted']
    flagged=[i for i,w in enumerate(rows) if w.get('source_warnings') or w.get('exact_training_record_match')]
    if key=='reliability':
        issue_ids={w['sample_id'] for w in read_rows(ROOT/'reports/reliability_workflow_audit/v1/ai_judgments.jsonl') if w['verdict']=='Issue'}
        flagged=sorted(set(flagged)|{i for i,w in enumerate(rows) if w['sample_id'] in issue_ids})
    return {'dataset':key,'label':CATALOG[key]['label'],'policy':policy,'chosen':chosen,'baseline':baseline,'flagged_indices':flagged,
        'baseline_label':baseline_label,'training_rows':len(data(key)['train']),'notes':notes,
        'records_path':str(records.relative_to(ROOT)),'records_sha256':sha(records),
        'metrics_path':str(metrics_file.relative_to(ROOT)),'metrics_sha256':sha(metrics_file),
        'adapter':adapter,'live_default_replaced':False}


def main():
    OUT.mkdir(parents=True,exist_ok=True);items={}
    for key in ['bitext','reliability']:
        directory=PREFIX/key;m=read_json(directory/'metrics.json');p=read_json(directory/'protocol.json')
        notes=['Category labels are supplied by an external assistant prefix; label-frequency gains are not learned probabilities.',
            'Fixed development cohort, including 8 reused pilot records. Remaining IDs 8 onward have separate metrics; no fresh final test.',
            'Copy matching normalizes case, punctuation and spacing. It does not certify privacy.']
        if key=='bitext':notes+=['All 39 intents covered. Numeric text-length distances improve, but request-length-bin TV and reply reuse do not both improve. One whole-record training match remains.']
        else:notes+=['Action-count, state and channel distributions regress despite improved category/route and message-length matching.',
            'Coherence review: 45 Pass, 15 Unsure, 4 Issue, out of 64. One unblinded AI reviewer; no expert calibration or factual-certification claim.']
        items[key]=make_item(key,m['arms']['controlled'],m['arms']['baseline'],'Source-label prefix · development',directory/'controlled_records.jsonl',directory/'metrics.json',notes,'Same adapter, generic prompt',p['adapter'])
    key='insuranceqa';directory=ROOT/CATALOG[key]['assessment'];m=read_json(directory/'assessment.json')
    selection=read_json(ROOT/CATALOG[key]['selection'])
    items[key]=make_item(key,m['labels']['tuned'],None,'Original fine-tuned generator · retained',directory/'tuned_records.jsonl',directory/'assessment.json',
        ['New conditioning attempts failed formatting gates, so retain the original generation method.',
         'This 128-output cohort is the original assessment, not a new experiment. Topic/length similarity does not establish insurance-fact correctness.',
         'Prior relevance review found 11 Issue and 4 Unsure among 30 stratified examples; this is an uncalibrated AI review, not an error-rate estimate.'],None,selection['adapter'])
    key='mimic_demo';m=read_json(CLINICAL/'pilot_metrics.json');selection=read_json(CLINICAL/'selection.json')
    items[key]=make_item(key,m['arms']['candidate_conditioned'],m['arms']['old_conditioned'],'Mixed-prompt candidate + diagnosis/null conditions · development',CLINICAL/'candidate_conditioned_records.jsonl',CLINICAL/'pilot_metrics.json',
        ['Comparison changes the saved adapter; both arms use the same diagnosis-pair and joint lab-null controls. Training budgets differ.',
         'Prefer this lighter controlled candidate for presentation: 64/64 supplied-condition compliance, zero observed prefix/version flags and one source-vocabulary race violation.',
         'The later full-profile controller improved numeric distances further but introduced two prefix/version flags and three source-constraint failures; it is not selected here.',
         'Categorical and correlation gaps remain. Prefix checking is not codebook validation, and no expert clinical plausibility review has been performed.',
         'Official 100-patient demo, 230 training admissions from 80 patients. Findings do not represent full MIMIC or the U.S. population. Candidate is not the live default.'],
        'Old adapter, identical diagnosis/null controls',selection['adapter'])
    review=read_json(ROOT/'reports/reliability_workflow_audit/v1/review_summary.json')
    snapshot={'scope':'Selected development evidence; no automatic deployment or new test scoring. Different sources use different cohorts/policies; no pooled accuracy score.',
        'datasets':items,'reliability_review':review,'report':'reports/assessment_snapshot/report.md'}
    final_path=ROOT/'outputs/reliability-consistency-v1/decision.json'
    final=read_json(final_path) if final_path.exists() else None
    if final:snapshot['final_reliability_attempt']={k:final[k] for k in ['decision','gates','review_counts','prior_issues_to_pass','numeric_wasserstein','limit']}
    write_json(OUT/'snapshot.json',snapshot)
    lines=['# Gemma synthetic-data assessment: current evidence','',snapshot['scope'],'',
        'The assignment asks whether generated data matches training ranges, distributions and characteristics. We measure these separately from formatting, novelty and semantic correctness. No universal pass threshold was supplied, and no test of exact distribution equivalence has been established.','',
        '| Dataset | Training rows | Selected policy | Valid / attempted | Source constraints | Whole-record match fraction |',
        '|---|---:|---|---:|---:|---:|']
    for key,item in items.items():
        a=item['chosen'];lines.append(f'| {item["label"]} | {item["training_rows"]} | {item["policy"]} | {a["structure_valid"]}/{a["attempted"]} | {a["source_constraint_valid"]}/{a["attempted"]} | {a["exact_training_record_match_rate"]} |')
    lines+=['','## What the criterion means in this project','',
        '- Ranges: min/max, quartiles and rates outside training support; text fields use character lengths. Staying inside a range alone does not show matching distributions.',
        '- Distribution: label TV, numeric KS/Wasserstein, missingness and clinical correlations. Source-imposed label/diagnosis/null frequencies are disclosed.',
        '- Characteristics: schema, supported label combinations, language markers, placeholders, histories/actions and request/reply coherence.',
        '- Novelty: normalized whole-record and field reuse. Template reuse and repeated finite workflow labels require source-specific interpretation; these checks do not certify privacy.']
    for key,item in items.items():
        a,b=item['chosen'],item['baseline'];lines+=['',f'## {item["label"]}','',*['- '+n for n in item['notes']],
            '',f'Frozen cohort: `{item["records_path"]}`. Metric provenance: `{item["metrics_path"]}`. Saved adapter: `{item["adapter"]}`.','']
        if b:
            lines+=['| Field | Wasserstein baseline | Selected |','|---|---:|---:|']
            for field,x in a['fidelity_vs_training']['numeric'].items():
                before=b['fidelity_vs_training']['numeric'][field].get('wasserstein');after=x.get('wasserstein')
                lines.append(f'| {field} | {before:.4g} | {after:.4g} |' if before is not None and after is not None else f'| {field} | Not estimable | Not estimable |')
        else:lines+=['Retained original assessment; no new before/after claim.']
        lines+=['','Observed ranges and missingness (schema-valid records; source is the training split):','',
            '| Field | Source min / median / max | Selected min / median / max | Missing source / selected | Selected outside source range |',
            '|---|---|---|---|---|']
        def triple(values):return ' / '.join(f'{values[k]:.4g}' for k in ['min','median','max']) if values else 'Not estimable'
        def percent(value):return 'Not estimable' if value is None else f'{100*value:.1f}%'
        for field,value in a['fidelity_vs_training']['numeric'].items():
            lines.append(f'| {field} | {triple(value.get("reference"))} | {triple(value.get("generated"))} | {percent(value.get("reference_missing_rate"))} / {percent(value.get("generated_missing_rate"))} | {percent(value.get("generated_outside_reference_range_rate"))} |')
        lines+=['','| Source characteristic | Baseline TV | Selected TV |','|---|---:|---:|']
        if a.get('feature_fidelity'):
            values={f:v['TV'] for f,v in a['feature_fidelity'].items()}
            before={f:v['TV'] for f,v in b['feature_fidelity'].items()} if b else {}
        else:
            values={f:v['total_variation'] for f,v in a['fidelity_vs_training']['categorical'].items()}
            before={f:v['total_variation'] for f,v in b['fidelity_vs_training']['categorical'].items()} if b else {}
        for field,value in values.items():
            old=before.get(field);lines.append(f'| {field} | {"—" if old is None else f"{old:.4f}"} | {"Not estimable" if value is None else f"{value:.4f}"} |')
    lines+=['','## Reliability: what needs fixing','',
        'All64 selected scenarios were reviewed by the assistant:45 Pass,15 Unsure,4 Issue. Generic baseline scenarios were not semantically scored, so no comparative semantic improvement is claimed. See `reports/reliability_workflow_audit/v1/report.md` for every judgment and source-support diagnostic.',
        'Nine workflows have fewer actions than their observed source subgroup;16 include an action label absent from training. These flags are not automatically errors. Do not pad action lists or replace them with source answers merely to improve a histogram. ID42 duplicates a payment already completed in history; ID18 misses an explicit deductible request; IDs33 and63 need policy-context reconciliation.',
        '', '## Delivery and remaining limits','',
        'The local demo exposes these cohorts and metrics as development evidence; original live model defaults remain unchanged. Earlier charts belong to the original assessments and must not be mixed with the newer cohorts.',
        'Remaining work for a stronger claim: independent seed confirmation; more original free-text variation; targeted workflow/history consistency improvement; insurance expert factual review; clinical/codebook validation for MIMIC. Broad retraining is not justified merely by one worsened marginal statistic. All prior failed experiments remain available in decision.md.']
    if final:lines+=['','## Final bounded attempt and freeze','',
        'One consistency-instruction variant was tested on four flagged and eight passing scenarios, with fixed inputs:24 total attempts. The guard was rejected. Schema validity12/12→11/12; no flagged case became a Pass; completed-payment repetition persists; action-count Wasserstein0.0805→0.1487. One passing control became Unsure. Workflow-length Wasserstein improved, which does not excuse the regressions. See `outputs/reliability-consistency-v1/report.md`. No more prompt/training experiments are planned before submission.']
    if (ROOT/'reports/criteria_evidence/v1/reference_bands.json').exists():
        lines+=['','## Criterion-facing visual evidence','',
            'Open `reports/criteria_evidence/v1/index.html` for offline source-versus-synthetic distribution charts, range tables, clinical relationship heatmaps and illustrative examples for all four datasets. The local frontend links to `/evidence/criteria`.',
            'The accompanying `report.md` and `reference_bands.json` add 200 equal-size training-subsample comparisons per dataset. Each sample is compared with its full training reference, as are synthetic scores. The 5th–95th percentile bands are descriptive finite-dataset sampling context, not confidence intervals or pass thresholds. Samples overlap the reference; clinical resampling is by admission, not patient. Externally controlled labels are not a matched unconditional sampling comparison. No new generation, selection, or test scoring occurred.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Current snapshot built:',OUT/'report.md',flush=True)


if __name__=='__main__':main()
