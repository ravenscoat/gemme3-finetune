"""Check immutable inputs, matched attempts and controller provenance after the pilot."""
import argparse
import json
from pathlib import Path

from dataset_catalog import ROOT,data,read_json,read_rows,sha
from json_io import write_json
from mimic_fidelity_pilot import OUT,N,missing_pattern
from advanced_exploratory_audits import icd_prefix_issue


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--conditions-only',action='store_true');args=parser.parse_args()
    if args.conditions_only:OUT=ROOT/'outputs/mimic-demo-fidelity-conditions-only-v1'
    assert (OUT/'completion.json').exists(),'Pilot has not completed'
    protocol=read_json(OUT/'protocol.json');result=read_json(OUT/'pilot_metrics.json')
    assert sha(OUT/'pilot_metrics.json')==read_json(OUT/'completion.json')['metrics_sha256']
    assert sha(OUT/'protocol.json')==result['protocol_sha256']==read_json(OUT/'baseline_freeze.json')['protocol_sha256']
    assert sha(Path(protocol['adapter'])/'adapter_model.safetensors')==protocol['adapter_sha256']
    for path,digest in protocol['source_hashes'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/protocol['baseline_source'])==protocol['baseline_source_sha256']
    assert read_rows(OUT/'baseline_generations.jsonl')==read_rows(ROOT/protocol['baseline_source'])[:N]
    supported={(r['primary_diagnosis_code'],r['primary_diagnosis_version'],missing_pattern(r)) for r in data('mimic_demo')['train']}
    for condition in protocol['conditions']:
        assert set(condition)=={'primary_diagnosis_code','primary_diagnosis_version','lab_missing_pattern'}
        assert (condition['primary_diagnosis_code'],condition['primary_diagnosis_version'],condition['lab_missing_pattern']) in supported
    checked={}
    for label,measure in result['arms'].items():
        raw=read_rows(OUT/(label+'_generations.jsonl'));rows=read_rows(OUT/(label+'_records.jsonl'))
        assert len(raw)==N and [r['sample_id'] for r in raw]==list(range(N))
        assert all(r['batch_seed']==protocol['batch_seed_start']+r['sample_id']//4 for r in raw)
        assert sha(OUT/(label+'_generations.jsonl'))==measure['raw_sha256']
        assert len(rows)==measure['structure_valid'] and len(rows)+len(measure['failures'])==N
        assert sum(icd_prefix_issue(r['record']) is not None for r in rows)==measure['prefix_version_flags']
        for row in rows:assert len(row['record'])==13
        checked[label]={'attempts':N,'valid':len(rows),'raw_hash_verified':True}
    assert set(checked)==set(protocol['arms'])
    conditional_label='condition_only' if args.conditions_only else 'conditional_rule'
    conditional=read_rows(OUT/(conditional_label+'_records.jsonl'));matches=[]
    for row in conditional:
        condition=protocol['conditions'][row['sample_id']];record=row['record']
        matches.append(record['primary_diagnosis_code']==condition['primary_diagnosis_code'] and record['primary_diagnosis_version']==condition['primary_diagnosis_version'] and missing_pattern(record)==condition['lab_missing_pattern'])
    assert sum(matches)==result['arms'][conditional_label]['condition_compliance']['all']
    assert not (ROOT/'reports/assessment_gpu.lock').exists(),'GPU lock was not released'
    verification={'status':'verified','arms':checked,'condition_provenance':'Every prescribed diagnosis pair/null pattern exists jointly in training',
        'adapter_unchanged':True,'source_hashes_unchanged':True,'baseline_exactly_matched':True,
        'gpu_reservation_released':True,'scope':'Integrity verification, not clinical certification or independent final testing'}
    write_json(OUT/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
