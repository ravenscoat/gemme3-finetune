"""Summarize the finished development comparison and standard table-quality checks."""
import importlib.metadata
import json
import numpy as np
import pandas as pd
from sdmetrics.reports import QualityReport

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha
from json_io import write_json
from mimic_conditioned_data import RUN


def main():
    metrics=read_json(RUN/'pilot_metrics.json');training=read_json(RUN/'training_summary.json')
    assert len(metrics['arms'])==4 and (RUN/'completion.json').exists()
    reference=pd.DataFrame([core('mimic_demo',r) for r in data('mimic_demo')['train']])
    metadata_path=ROOT/'reports/advanced_analysis/v1/mimic_metadata.json';metadata=read_json(metadata_path)
    quality={'version':importlib.metadata.version('sdmetrics'),'reference_admissions':230,'metadata_sha256':sha(metadata_path),
             'comparison_metrics_sha256':sha(RUN/'pilot_metrics.json'),'arms':{},
             'limits':'Statistical similarity, not clinical accuracy.64 attempts per arm; source constraints and failures remain separate. No independent final-test claim.'}
    for label,m in metrics['arms'].items():
        rows=read_rows(RUN/(label+'_records.jsonl'))
        if not rows:quality['arms'][label]={'n':0,'quality':None};continue
        np.random.seed(2026110106)
        report=QualityReport();report.generate({'admissions':reference},{'admissions':pd.DataFrame([r['record'] for r in rows])},metadata,verbose=False)
        quality['arms'][label]={'n':len(rows),'quality':float(report.get_score()),'properties':json.loads(report.get_properties().to_json(orient='records'))}
    write_json(RUN/'sdmetrics_comparison.json',quality)
    lines=['# MIMIC mixed-prompt fine-tuning: results','',
        'A separate Gemma3 4B QLoRA candidate was trained and evaluated. All original adapters remain unchanged. The candidate is saved and **not deployed**.','',
        f"Training:460 prompt views of230 admissions from80 patients,2 epochs /116 steps. Selected checkpoint:{training['selection']['best_checkpoint'].split('checkpoint-')[-1]}. Runtime including preparation:{training['elapsed_seconds']/60:.1f}minutes; peak allocated GPU:{training['peak_gpu_allocated_gib']:.2f}GiB.",'',
        f"Generic validation completion loss:{training['validation_before']['eval_generic_loss']:.4f}→{training['validation_after']['eval_generic_loss']:.4f}. Old saved model:0.3264. Selection uses21 validation admissions from10 separate patients; no new test scoring.",'',
        'Each original admission has one generic and one condition-specific prompt view, with unchanged target values/order. This adds no new patients. Original run budget230x3 versus candidate460x2; this is not a compute-matched causal ablation.','',
        '| Method | Structure /64 | Source constraints /64 | ICD-prefix flags | Whole-record copies | SDMetrics quality |','|---|---:|---:|---:|---:|---:|']
    for label,m in metrics['arms'].items():
        copy=m['exact_training_record_match_rate'];copy='n/a' if copy is None else f'{100*copy:.1f}%'
        q=quality['arms'][label]['quality'];q='n/a' if q is None else f'{q:.3f}'
        lines.append(f"| {label} | {m['structure_valid']} | {m['source_constraint_valid']} | {m['prefix_version_flags']} | {copy} | {q} |")
    lines+=['','## Changes under the same generation method','',
            'Wasserstein distances below retain each field’s units; compare old versus candidate within a field, not across different units. Lower is closer.','',
            '| Field | Old generic | Candidate generic | Old conditioned | Candidate conditioned |','|---|---:|---:|---:|---:|']
    fields=data('mimic_demo')['schema']['numeric_fields']
    labels=['old_generic','candidate_generic','old_conditioned','candidate_conditioned']
    for field in fields:
        values=[metrics['arms'][label]['fidelity_vs_training']['numeric'][field].get('wasserstein') for label in labels]
        lines.append('| '+field+' | '+' | '.join('n/a' if v is None else f'{v:.3f}' for v in values)+' |')
    lines+=['','## Missing labs','',
            '| Field | Source missing | Old generic | Candidate generic | Old conditioned | Candidate conditioned |','|---|---:|---:|---:|---:|---:|']
    for field in fields[2:]:
        values=[metrics['arms'][label]['fidelity_vs_training']['numeric'][field]['generated_missing_rate'] for label in labels]
        source=metrics['arms']['old_generic']['fidelity_vs_training']['numeric'][field]['reference_missing_rate']
        lines.append('| '+field+f' | {100*source:.1f}% | '+' | '.join('n/a' if v is None else f'{100*v:.1f}%' for v in values)+' |')
    lines+=['','## Interpretation and limitations','',
        '- Generic generation tests the original prompt. Conditioned generation uses the exact same diagnosis pairs and lab-null patterns for both models, chosen from training before generation.',
        '- Supplied conditions are externally controlled. Good compliance is not proof that Gemma learned their probabilities.',
        '- All64 attempts per arm remain in validity denominators. No retries, repair or selective replacement. Full-record copying is checked against training.',
        '- Read categorical TV, numeric ranges/quantiles, outside-range rates and correlation differences in `pilot_metrics.json`; an overall score can hide a worse field.',
        '- SDMetrics is statistical similarity, not accuracy. The prefix screen is not full diagnosis-code or clinical validation.',
        '- These reused-seed development samples are small. Previously used final tests were not rescored, and no new independent final-test claim is made.',
        '- Baselines, conditions, prompts, source hashes and sampling settings are fixed in `comparison_plan.json`. The selected candidate hash is frozen in `generation_protocol.json`.',
        '', 'Interactive offline plots: [generic distributions](distributions_generic.html), [conditioned distributions](distributions_conditioned.html).',
        '', 'Integrity evidence: `verification.json`. Full source/prompt views: `data/mimic_demo/conditioned_v1/manifest.json`.','']
    (RUN/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved standard quality checks and consolidated candidate report.')


if __name__=='__main__':main()
