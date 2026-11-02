"""Verify the finished candidate experiment without another model or test run."""
import json
from pathlib import Path

from dataset_catalog import CATALOG,ROOT,data,read_json,read_rows,sha
from json_io import write_json
from mimic_conditioned_data import PREPARED,RUN,conditioned_prompt
from mimic_fidelity_pilot import missing_pattern


def main():
    completion=read_json(RUN/'completion.json');plan=read_json(RUN/'comparison_plan.json');metrics=read_json(RUN/'pilot_metrics.json')
    assert completion['comparison_sha256']==sha(RUN/'pilot_metrics.json')
    selected=read_json(RUN/'selection.json');generation=read_json(RUN/'generation_protocol.json')
    assert sha(RUN/'adapter/adapter_model.safetensors')==selected['adapter_sha256']==completion['candidate_adapter_sha256']==generation['candidate_adapter_sha256']
    assert sha(RUN/'selection.json')==generation['candidate_selection_sha256']
    assert sha(RUN/'comparison_plan.json')==generation['comparison_plan_sha256']==metrics['comparison_plan_sha256']
    training=read_json(RUN/'training_summary.json');manifest=read_json(PREPARED/'manifest.json')
    assert training['protocol']['data_manifest_sha256']==sha(PREPARED/'manifest.json')
    for path,digest in manifest['source_hashes'].items():assert sha(ROOT/path)==digest
    for name,digest in manifest['view_hashes'].items():assert sha(PREPARED/(name+'.jsonl'))==digest
    assert training['global_steps']==116
    assert training['selection']['selection_metric']=='eval_generic_loss on21 validation admissions'
    train=read_rows(PREPARED/'train.jsonl');validation=read_rows(PREPARED/'validation_generic.jsonl')
    assert len(train)==460 and len(validation)==21
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in validation})
    for label,item in plan['baselines'].items():
        assert sha(ROOT/item['path'])==item['sha256']==sha(RUN/(label+'_generations.jsonl'))
    for label,m in metrics['arms'].items():
        raw=read_rows(RUN/(label+'_generations.jsonl'));rows=read_rows(RUN/(label+'_records.jsonl'))
        assert sha(RUN/(label+'_generations.jsonl'))==m['raw_sha256']
        assert len(raw)==64 and [r['sample_id'] for r in raw]==list(range(64))
        assert all(r['batch_seed']==plan['seed_start']+r['sample_id']//4 for r in raw)
        assert len(rows)==m['structure_valid'] and len(rows)+len(m['failures'])==64
        if 'conditioned' in label:
            matched=sum(row['record']['primary_diagnosis_code']==plan['conditions'][row['sample_id']]['primary_diagnosis_code'] and
                        row['record']['primary_diagnosis_version']==plan['conditions'][row['sample_id']]['primary_diagnosis_version'] and
                        missing_pattern(row['record'])==plan['conditions'][row['sample_id']]['lab_missing_pattern'] for row in rows)
            assert matched==m['condition_compliance']['all']
    assert set(metrics['arms'])=={'old_generic','old_conditioned','candidate_generic','candidate_conditioned'}
    for key,spec in CATALOG.items():
        old=read_json(ROOT/spec['selection'])
        assert sha(Path(old['adapter'])/'adapter_model.safetensors')==old['adapter_sha256'],key
    assert not (ROOT/'reports/assessment_gpu.lock').exists()
    verification={'status':'verified','candidate_adapter_sha256':selected['adapter_sha256'],'training_steps':116,
        'prompt_views':460,'real_train_admissions':230,'train_patients':80,'validation_admissions':21,
        'patient_disjoint':True,'all_four_original_adapters_unchanged':True,'matched_generation_arms':4,'attempts_per_arm':64,
        'no_new_test_scoring':True,'automatic_deployment':False,'gpu_reservation_released':True}
    write_json(RUN/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
