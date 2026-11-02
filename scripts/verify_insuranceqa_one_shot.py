"""Verify the one-demo intervention and independently check saved score arithmetic."""
import math
import json
from dataset_catalog import ROOT,read_json,read_rows,sha
from insuranceqa_one_shot_pilot import OUT,PRIOR,DEMONSTRATION
import insuranceqa_final_pilot as engine
import verify_insuranceqa_final_pilot as common
from json_io import write_json

def main():
    engine.OUT=OUT;common.OUT=OUT;common.check=engine.check;common.main()
    p=read_json(OUT/'protocol.json');old=read_json(PRIOR/'protocol.json')
    for field in ['anchors','adapter','context','batch_size','temperature','top_p','top_k','seed_start','domain_quotas']:
        assert p[field]==old[field],field
    assert p['prompts']==[DEMONSTRATION+x for x in old['prompts']]
    assert sha(OUT/'baseline_generations.jsonl')==sha(PRIOR/'grounded_generations.jsonl')
    a=read_json(OUT/'metrics.json');b=read_json(PRIOR/'metrics.json')
    assert math.isclose(a['arms']['baseline']['sdmetrics']['quality'],b['arms']['grounded']['sdmetrics']['quality'],abs_tol=1e-12)
    for view in ['request','reply','joint']:
        assert math.isclose(a['text_metrics'][view]['arms']['baseline']['mmd2'],b['text_metrics'][view]['arms']['grounded']['mmd2'],abs_tol=1e-12)
    expected={r['sample_id']:r for r in read_rows(PRIOR/'judgments.jsonl') if r['arm']=='grounded'}
    for review in read_rows(OUT/'judgments.jsonl'):
        if review['arm']=='baseline':
            for field in ['record_sha256','meaning_verdict','meaning_reason','source_support_verdict','source_support_reason']:
                assert review[field]==expected[review['sample_id']][field]
    manifest=read_json(OUT/'embedding_manifest.json')['features']['insuranceqa']
    prior_manifest=read_json(PRIOR/'embedding_manifest.json')['features']['insuranceqa']
    for view in ['request','reply','joint']:
        assert manifest['baseline'][view]['sha256']==prior_manifest['grounded'][view]['sha256']
    result=read_json(OUT/'verification.json');result.update(single_added_demonstration_verified=True,all32_source_anchors_identical=True,
        frozen_baseline_judgments_reused_without_changes=True,baseline_scores_and_features_match_prior=True,
        software={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'scripts/insuranceqa_one_shot_pilot.py',ROOT/'scripts/review_insuranceqa_one_shot.py',ROOT/'scripts/verify_insuranceqa_one_shot.py']})
    write_json(OUT/'verification.json',result);print('Single-change protocol, identical anchors and unchanged baseline review/features verified.')

if __name__=='__main__':main()
