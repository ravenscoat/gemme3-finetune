"""Independent final-pilot arithmetic, immutable-input and review checks."""
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from dataset_catalog import ROOT,read_json,read_rows,sha
from insuranceqa_final_pilot import OUT,check
from json_io import write_json

def main():
    check();m=read_json(OUT/'metrics.json');manifest=read_json(OUT/'embedding_manifest.json')
    assert manifest['input_sha256']==sha(OUT/'embedding_inputs.json')
    assert read_json(OUT/'generation_complete.json')['raw_sha256']==sha(OUT/'grounded_generations.jsonl')
    recalcs=0
    for view,scores in m['text_metrics'].items():
        arrays={}
        for arm,info in manifest['features']['insuranceqa'].items():
            path=ROOT/info[view]['path'];assert sha(path)==info[view]['sha256']
            arrays[arm]=np.load(path).astype(float)
            assert np.max(np.abs(np.linalg.norm(arrays[arm],axis=1)-1))<1e-5
        x=arrays['training_reference'];gamma=scores['gamma']
        for arm,value in scores['arms'].items():
            y=arrays[arm];actual=max(0.,float(rbf_kernel(x,gamma=gamma).mean()+rbf_kernel(y,gamma=gamma).mean()-2*rbf_kernel(x,y,gamma=gamma).mean()))
            assert abs(actual-value['mmd2'])<1e-9;recalcs+=1
    for arm,value in m['keywords']['collections'].items():
        table=pd.read_csv(OUT/'insuranceqa'/(arm+'_keywords.csv'));p=table.source_share.to_numpy();q=table.generated_share.to_numpy();mid=(p+q)/2
        def kl(a,b):
            mask=a>0;return float(np.sum(a[mask]*np.log2(a[mask]/b[mask])))
        assert abs(math.sqrt(max(0.,(kl(p,mid)+kl(q,mid))/2))-value['js_distance_base2'])<1e-9
    judgments=read_rows(OUT/'judgments.jsonl')
    for arm,value in m['arms'].items():
        records=read_rows(OUT/(arm+'_records.jsonl'));assert len(records)==value['structure_valid']
        review=[r for r in judgments if r['arm']==arm];assert len(review)==len(records)
        from dataset_catalog import signature
        import hashlib
        for r,j in zip(records,review):
            assert r['sample_id']==j['sample_id']
            assert hashlib.sha256(signature(r['record']).encode()).hexdigest()==j['record_sha256']
    freeze=read_json(ROOT/'reports/final_experiment_freeze.json');originals=0
    for adapter in freeze['adapters'].values():
        from pathlib import Path
        for file,field in [('adapter_model.safetensors','weight_sha256'),('adapter_config.json','config_sha256')]:
            assert sha(Path(adapter['original_path'])/file)==adapter[field];originals+=1
    # Check original cohorts through the previously frozen benchmark input hashes.
    original_protocol=read_json(ROOT/'reports/distribution_benchmark/v1/protocol.json')
    for path,digest in original_protocol['input_hashes'].items():assert sha(ROOT/path)==digest
    snapshot=read_json(ROOT/'reports/assessment_snapshot/snapshot.json')['datasets']
    for key,cohort in freeze['selected_cohorts'].items():
        for field,hashfield in [('records_path','records_sha256'),('metrics_path','metrics_sha256')]:
            assert sha(ROOT/snapshot[key][field])==cohort[hashfield]
            originals+=1
    for path in OUT.glob('*.json'):
        json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
    assert not (ROOT/'reports/assessment_gpu.lock').exists()
    audit={'status':'passed','independent_mmd_recalculations':recalcs,'independent_keyword_recalculations':2,
       'all_valid_records_reviewed':True,'original_model_and_cohort_files_verified':originals,'original_benchmark_inputs_unchanged':True,
       'frozen_protocol_and_source_verified':True,'gpu_lock_released':True,
       'artifacts':{str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='verification.json'}}
    write_json(OUT/'verification.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='artifacts'},indent=2))

if __name__=='__main__':main()
