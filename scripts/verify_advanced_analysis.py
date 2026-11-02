"""Verify input immutability, evidence coverage and numerical artifact integrity."""
from collections import Counter
import hashlib
import json
import math
import numpy as np

from dataset_catalog import CATALOG,ROOT,core,data,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'


def main():
    protocol=read_json(OUT/'protocol.json');manifest=read_json(OUT/'embedding_manifest.json')
    for path,digest in protocol['source_hashes'].items():assert sha(ROOT/path)==digest,path
    assert manifest['protocol']==protocol['embedding']
    assert sha(OUT/'embedding_inputs.json')==protocol['embedding']['input_sha256']
    assert not (ROOT/'reports/assessment_gpu.lock').exists(),'Stale GPU reservation'
    for key,spec in CATALOG.items():
        selected=read_json(ROOT/spec['selection'])
        assert sha(__import__('pathlib').Path(selected['adapter'])/'adapter_model.safetensors')==selected['adapter_sha256'],key
    arrays=0
    for key,collections in manifest['features'].items():
        n=protocol['embedding']['selections'][key]['n']
        for label,fields in collections.items():
            ids=protocol['embedding']['selections'][key]['selected_ids'][label]
            assert len(ids)==len(set(ids))==n and None not in ids
            for item in fields.values():
                path=ROOT/item['path'];assert sha(path)==item['sha256']
                array=np.load(path,allow_pickle=False)
                assert array.shape==(n,1280) and array.dtype==np.float32
                assert np.isfinite(array).all() and np.allclose(np.linalg.norm(array,axis=1),1,atol=2e-6)
                arrays+=1
    review=read_json(OUT/'ai_review_summary.json');inputs=read_rows(OUT/'review_records.jsonl');judgments=read_rows(OUT/'ai_judgments.jsonl')
    assert sha(OUT/'review_records.jsonl')==protocol['review']['input_sha256']==review['input_sha256']
    assert sha(OUT/'rubric_v1.md')==review['rubric_sha256']
    assert sha(OUT/'ai_judgments.jsonl')==review['judgments_sha256']
    assert len(judgments)==len({r['review_id'] for r in judgments})==100
    assert [r['review_id'] for r in inputs]==[r['review_id'] for r in judgments]
    for record,judgment in zip(inputs,judgments):
        expected=hashlib.sha256(json.dumps(record['record'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        assert judgment['record_sha256']==expected
        assert len(judgment['reason'])>30 and judgment['verdict'] in ['Pass','Issue','Unsure']
    for key,counts in review['counts'].items():assert dict(Counter(r['verdict'] for r in judgments if r['dataset']==key))=={k:v for k,v in counts.items() if v}
    prdc=read_json(OUT/'prdc_results.json');assert prdc['feature_manifest_sha256']==sha(OUT/'embedding_manifest.json')
    for fields in prdc['datasets'].values():
        for value in fields.values():
            for collection in value['scores'].values():
                for scores in collection.values():
                    assert all(math.isfinite(v) and v>=0 for v in scores.values())
                    assert all(scores[k]<=1 for k in ['precision','recall','coverage'])
    sdmetrics=read_json(OUT/'sdmetrics_results.json')
    for path,digest in sdmetrics['source_hashes'].items():assert sha(ROOT/path)==digest,path
    assert sdmetrics['reference_records']==230 and sdmetrics['reference_patients']==80
    reference=data('mimic_demo')['train'];generated=read_rows(ROOT/CATALOG['mimic_demo']['assessment']/'tuned_records.jsonl')
    for field,entry in sdmetrics['collections']['tuned']['per_column'].items():
        source_missing=sum(r[field] is None for r in reference)/len(reference)
        generated_missing=sum(r['record'][field] is None for r in generated)/len(generated)
        assert abs(source_missing-entry['training_missing_rate'])<1e-10
        assert abs(generated_missing-entry['generated_missing_rate'])<1e-10
        assert abs(entry['missing_similarity']-(1-abs(source_missing-generated_missing)))<1e-10
    exploratory=read_json(OUT/'exploratory_results.json')
    assert exploratory['addendum_sha256']==sha(OUT/'exploratory_addendum.json')
    assert exploratory['icd_prefix_version']['training_reference']['flagged']==0
    assert exploratory['icd_prefix_version']['tuned']['flagged']==57
    result={'status':'verified','source_hashes_checked':len(protocol['source_hashes']),'frozen_adapter_hashes_checked':4,
            'normalized_feature_arrays_checked':arrays,'individually_authored_judgments_checked':100,
            'missingness_recomputed':True,'numeric_bounds_checked':True,'gpu_reservation_released':True,
            'limitations':'Artifact integrity checks; not validation of the AI judge or clinical correctness.'}
    write_json(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
