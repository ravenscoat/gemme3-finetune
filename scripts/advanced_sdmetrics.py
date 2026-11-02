"""Standard SDMetrics checks on frozen MIMIC demo outputs; no missing-value imputation."""
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sdmetrics.reports import QualityReport
from sdmetrics.single_column import MissingValueSimilarity,RangeCoverage,CategoryCoverage,BoundaryAdherence,CategoryAdherence

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'


def main():
    protocol=read_json(OUT/'protocol.json')
    for path,digest in protocol['source_hashes'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen advanced source changed: '+path)
    source=data('mimic_demo');fields=source['schema']['fields'];numeric=source['schema']['numeric_fields']
    reference=pd.DataFrame([core('mimic_demo',r) for r in source['train']])
    columns={field:{'sdtype':'numerical' if field in numeric else 'boolean' if field=='age_topcoded' else 'categorical'} for field in fields}
    metadata={'tables':{'admissions':{'columns':columns}},'relationships':[]}
    write_json(OUT/'mimic_metadata.json',metadata)
    paths={'base':ROOT/source['spec']['assessment']/'base_records.jsonl',
           'tuned':ROOT/source['spec']['assessment']/'tuned_records.jsonl'}
    collections={k:pd.DataFrame([r['record'] for r in read_rows(path)]) for k,path in paths.items()}
    collections['validation_control']=pd.DataFrame([core('mimic_demo',r) for r in read_rows(ROOT/source['spec']['prepared']/'validation.jsonl')])
    result={'sdmetrics_version':importlib.metadata.version('sdmetrics'),'metadata_sha256':sha(OUT/'mimic_metadata.json'),
            'reference_records':len(reference),'reference_patients':len({r['group_id'] for r in source['train']}),
            'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths.values()},'collections':{},
            'limits':['Descriptive follow-up on frozen structurally valid outputs; original127/128, tuned128/128.',
                      'Quality averages are statistical similarities, not clinical accuracy or privacy guarantees.',
                      'The validation control has21 admissions from10 patients, unlike the128 generated attempts; scores are not directly sample-size matched.',
                      'Observed training ranges are not clinical normal/safety limits.',
                      'Missing values are retained asNaN; no rows are imputed or removed from the input tables.',
                      'Small single-site100-patient demo; multiple admissions per patient are dependent.']}
    lab_fields=['creatinine_mg_dl','glucose_mg_dl','hemoglobin_g_dl']
    def pattern_counts(frame):return Counter(''.join('1' if pd.isna(row[f]) else '0' for f in lab_fields) for _,row in frame.iterrows())
    ref_patterns=pattern_counts(reference)
    for label,frame in collections.items():
        report=QualityReport();report.generate({'admissions':reference},{'admissions':frame},metadata,verbose=False)
        properties=json.loads(report.get_properties().to_json(orient='records'))
        details={name:json.loads(report.get_details(name).to_json(orient='records')) for name in ['Column Shapes','Column Pair Trends']}
        per_column={}
        for field in fields:
            a,b=reference[field],frame[field]
            entry={'missing_similarity':float(MissingValueSimilarity.compute(a,b)),
                   'training_missing_rate':float(a.isna().mean()),'generated_missing_rate':float(b.isna().mean()),
                   'absolute_missing_percentage_point_error':100*abs(float(a.isna().mean())-float(b.isna().mean()))}
            if field in numeric:
                entry.update(range_coverage=float(RangeCoverage.compute(a,b)),boundary_adherence=float(BoundaryAdherence.compute(a,b)))
            else:
                entry.update(category_coverage=float(CategoryCoverage.compute(a,b)),category_adherence=float(CategoryAdherence.compute(a,b)))
            per_column[field]=entry
        patterns=pattern_counts(frame)
        pattern_tv=.5*sum(abs(ref_patterns[p]/len(reference)-patterns[p]/len(frame)) for p in ref_patterns.keys()|patterns.keys())
        result['collections'][label]={'n':len(frame),'overall_quality':float(report.get_score()),'properties':properties,'details':details,
            'per_column':per_column,'lab_missing_patterns':{'field_order':lab_fields,'training_counts':dict(ref_patterns),'generated_counts':dict(patterns),'total_variation':pattern_tv}}
        if label=='tuned':
            for name in ['Column Shapes','Column Pair Trends']:
                report.get_visualization(name,table_name='admissions').write_html(OUT/('mimic_'+name.lower().replace(' ','_')+'.html'),include_plotlyjs=True)
        print(label,'SDMetrics quality',round(report.get_score(),4),'n',len(frame),flush=True)
    write_json(OUT/'sdmetrics_results.json',result)
    lines=['# MIMIC demo: SDMetrics follow-up','',f"SDMetrics {result['sdmetrics_version']}; reference230 training admissions. Missing values retained.",'',
           '| Collection | Records | Overall statistical quality |','|---|---:|---:|']
    for label,r in result['collections'].items():lines.append(f"| {label} | {r['n']} | {r['overall_quality']:.4f} |")
    lines+=['','## Missingness and coverage: tuned output','',
            '| Field | Source missing | Generated missing | Absolute gap, percentage points | Range/category coverage |','|---|---:|---:|---:|---:|']
    for field,r in result['collections']['tuned']['per_column'].items():lines.append(f"| {field} | {100*r['training_missing_rate']:.1f}% | {100*r['generated_missing_rate']:.1f}% | {r['absolute_missing_percentage_point_error']:.2f} | {r.get('range_coverage',r.get('category_coverage')):.3f} |")
    lines+=['','A high missing-value similarity can coexist with a material relative shortfall in missing labs. Read rates and percentage-point gaps, not only the aggregate.','',*['- '+v for v in result['limits']],'']
    (OUT/'sdmetrics_report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
